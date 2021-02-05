"""
created by haoran
time:2021/01/26
description:
1. 用来生成关键点定位网络的数据
"""
import numpy as np
import pandas as pd
import os
import cv2 as cv
from PIL import Image
import hashlib
import matplotlib.pyplot as plt
import traceback
import sys, time
import re
import rich


class LandMarkData:
    def __init__(self):
        self.save_path = 'H:/手掌关键点定位/after_resize2'
        self.finish_path = 'H:/手掌关键点定位/after_resize2'
        self.error_list = []
        # self.multi_landmark_path = self.multi_batch()
        # image_and_landmark = []
        # for idx, item in enumerate(self.multi_landmark_path):
        #     print('process: %d / %d' % (idx, len(self.multi_landmark_path)))
        #     image_and_landmark.append(self.read_csv(item))

    def multi_batch(self):
        """
        每个批次的数据，逐一处理，读取CSV在这里
        :return:
        """
        multi_landmark_path = []
        multi_image_path = []
        for i in range(1, 10):
            print('In multi_batch process: %d / 9' % (i))
            multi_landmark_path.append('./landmark_gt/第%d批check.csv' % (i))
            multi_image_path.append('H:/手掌关键点定位/第二大批原始数据/第%d批' % (i))
            _image_and_landmark_list = self.read_csv(multi_landmark_path[i - 1])
            _image_dir = multi_image_path[i - 1]
            self.deal_with_all_image(_image_and_landmark_list, image_dir=_image_dir)

        print('The number of error image is : %d' % (len(self.error_list)))
        print(self.error_list)

    def deal_with_one_image(self, image_and_landmark_dict=None, image_dir=None, size=192):
        """
        :param image_and_landmark_dict: 输入的是一个字典，只包含单张图片
        :param image_dir:输入的是图片所在的根目录，因为字典中只包含图片的名称
        :param size:希望将图片resize的大小
        :return:
        """
        try:
            # print(os.path.join(image_dir, image_and_landmark_dict['img_name']))
            src = Image.open(os.path.join(image_dir, image_and_landmark_dict['img_name']))
            if len(src.split()) == 3:
                pass
            elif len(src.split()) == 4:
                # convert
                src = src.convert("RGB")
            else:
                return False

        except Exception as e:
            print(e)
            return False

        # 检查gt
        # self.visualize(src, image_and_landmark_dict['landmark'])
        crop_result = self.palm_crop(image_and_landmark_dict['landmark'])

        palm_center_img, crop_landmark = self.crop_img(img=src,max_point=crop_result,
                                                      landmark = image_and_landmark_dict['landmark'],pad=100)

        # self.visualize(palm_center_img, crop_landmark)
        resize_landmark = self.point_map(landmark=crop_landmark, src_size=palm_center_img.size,
                                         src_resize_size=(size, size))
        palm_center_img_192 = palm_center_img.resize((size, size))
        # self.visualize(palm_center_img_192, resize_landmark)
        # self.draw_bbox(src_192, resize_landmark)

        # resize to 192
        # src_192 = src.resize((size, size))
        # self.visualize(src_192, resize_landmark)
        try:
            self.save_src_and_gt(src_192=palm_center_img_192, landmark=resize_landmark,
                                 img_name=image_and_landmark_dict['img_name'])
            pass
        except Exception as e:
            print(e)
            return False
        return True
    def crop_img(self, img, max_point, landmark, pad=20):
        """
        :param img: PIL IMAGE
        :param pad:
        :return:
        """
        img = np.array(img)
        top = max_point['max_top'] - pad if max_point['max_top'] - pad >=0 else 0
        botton = max_point['max_botton'] + pad if max_point['max_botton'] + pad >= 0 else img.size[0]
        left = max_point['max_left'] - pad if max_point['max_left'] - pad >= 0 else 0
        right = max_point['max_right'] + pad if max_point['max_right'] + pad >= 0 else img.size[1]
        top_gap = top
        print(img.shape)
        botton_gap = img.shape[0] - botton
        left_gap = left
        right_gap = img.shape[1] - right

        # correct landmark
        for idx, item in enumerate(landmark):
            landmark[idx] = (int(item[0]) - top_gap,int(item[1]) - left_gap)


        img = img[top:botton,left:right,:]
        img = np.array(img,dtype='uint8')
        img = Image.fromarray(img)
        return img,landmark

    def point_map(self, landmark, src_size, src_resize_size):
        """
        点映射，计算resize之后的坐标点
        :param landmark:
        :return:
        """
        # 缩小的倍数
        row_rate = src_size[0] / src_resize_size[0]
        col_rate = src_size[1] / src_resize_size[1]

        # 开始计算缩小后的坐标点的位置
        x, y = self.unzip(landmark=landmark)
        for i in range(12):
            x[i] = round(x[i] / col_rate,2)
            y[i] = round(y[i] / row_rate,2)

        return list(zip(x, y))

    def read_csv(self, landmark_path):
        """
        读取CSV文件，转化为易于处理的字典形式:
        但是这里有个小的问题：元组中数字类型的问题
        {'img_name': '20001-女-27-左.jpg',
        'landmark': [('2857', 1310.0), ('2204', 966.0), ('1776', 1160.0),
        ('1330', 1335.0), ('1238', 1599.0), ('1208', 1664.0),
         ('1239', 1903.0), ('1242', 1957.0), ('1388', 2193.0),
         ('1441', 2243.0), ('1645', 2474.0), ('2910', 2221.0)]}

        :return:
        """
        # read landmark file
        df = pd.read_csv(landmark_path, encoding='gb2312', header=None)
        print(df.shape[0])
        # the image name index = 0 13 26;
        name_index = [i * 13 for i in range(int(df.shape[0] / 13))]
        image_and_landmark = []
        for idx, item in enumerate(name_index):
            x = list(df.loc[[item + 1, item + 2, item + 3, item + 4, item + 5, item + 6, item + 7, item + 8, item + 9,
                             item + 10, item + 11, item + 12]][0])
            y = list(df.loc[[item + 1, item + 2, item + 3, item + 4, item + 5, item + 6, item + 7, item + 8, item + 9,
                             item + 10, item + 11, item + 12]][1])
            # for idx,item in enumerate(x):
            #     x[idx] = int(item)
            # for idx, item in enumerate(y):
            #     y[idx] = int(item)

            loc = list(zip(x, y))

            img_landmark_dict = {'img_name': str(df.loc[item][0]),
                                 'landmark': loc}
            image_and_landmark.append(img_landmark_dict)

        return image_and_landmark

    def save_src_and_gt(self, src_192, landmark, img_name):
        """
        这个方法使用来保存生成后的图片，landmark的坐标保存在图片命名中
        如21130-女-28-右;111,111-111,111-111,111-111,111-111,111-111,111-111,111-111,111-111,111-111,111-111,111-111,111-111
        :param src_192:PIL IMAGE
        :param landmark:包含12个元组的list
        :return:
        """
        # 首先检查保存的路径
        if not os.path.exists(self.save_path):
            traceback.print_exc('please enter legal save_path')

        # 生成保存的文件名
        name_name = img_name.split('.')[0]
        name_format = img_name.split('.')[-1]
        name_landmark = ''
        for item in landmark:
            name_landmark += '%.2f,%.2f-' % (item[0], item[1])

        save_name = name_name + ';' + name_landmark + '.' + name_format
        print(save_name)
        try:
            src_192.save(os.path.join(self.save_path, save_name))
        except Exception as e:
            print(e, 'the image :%s error' % save_name)

    def palm_crop(self, landmark, pad=10):
        """
        通过最值点crop确定框的大小
        :return:

        """
        max_top, max_botton, max_left, max_right = 9999, -1, 9999, -1
        for idx, item in enumerate(landmark):
            if int(item[0]) > max_botton:
                max_botton = int(item[0])
            elif int(item[0]) < max_top:
                max_top = int(item[0])
            else:
                pass

            if int(item[1]) > max_right:
                max_right = int(item[1])
            elif int(item[1]) < max_left:
                max_left = int(item[1])
            else:
                pass
        return {'max_top':max_top,
                'max_botton':max_botton,
                'max_left':max_left,
                'max_right':max_right}

    def draw_bbox(self, img, max_point):
        """

        :param img: PIL IMAGE
        :param max_point:
        :return:
        """
        print(max_point)
        img = np.array(img,dtype='uint8')
        plt.imshow(img)
        plt.gca().add_patch(plt.Rectangle(xy=(int(max_point['max_left']), int(max_point['max_top'])),
                                          width=max_point['max_right'] - max_point['max_left'],
                                          height=max_point['max_botton'] - max_point['max_top'],
                                          fill=False, linewidth=2))
        plt.show()

    def deal_with_all_image(self, image_and_landmark_list=None, image_dir=None):
        """
        这个方法是用来批量一批数据的
        :param image_and_landmark_list: 包含多个字典的list
        :param image_dir:这个list对应CSV文件所在的图片的目录
        :return:
        """

        # 错误判断
        if image_and_landmark_list == None or image_dir == None:
            traceback.print_exc('image_and_landmark_list or image_dir is None, please check input')

        # 开始处理列表中的每个字典
        for idx, item in enumerate(image_and_landmark_list):
            print('In deal_with_all_image function : %d / %d ' % (idx, len(image_and_landmark_list)))
            if self.deal_with_one_image(item, image_dir=image_dir):
                pass
            else:
                self.error_list.append(os.path.join(image_dir, item))

    def read_and_check(self):
        """
        读取处理好的图片，并且检查gt
        :return:
        """
        error_list = []
        # 0 read img
        if not os.path.exists(self.finish_path):
            traceback.print_exc('The input path error')

        for idx, item in enumerate(os.listdir(self.finish_path)):

            try:
                parse_result = self.parse_image_name(item)
            except:
                print(item)
            if parse_result == False:
                error_list.append(item)
            # if item.split('.')[0] == '20500-男-20-右;127,174-168,121-151,111-148,110-129,101-124,100-105,98-96,98-72,100-49,123-31,142-59,166-':
            src = Image.open(os.path.join(self.finish_path, item))
            # print(parse_result['landmark'])
            # self.visualize(src, landmark=parse_result['landmark'])
            # time.sleep(1)

    def visualize(self, img, landmark):
        """
        :param img: PIL Image 类型
        :param landmark: 一个list 其中每个点都是一个元组
        :return: 直接可视化结果
        """
        img = np.array(img, dtype='uint8')
        plt.figure('visualize')
        plt.imshow(img)
        x, y = self.unzip(landmark=landmark)
        plt.plot(y, x, '*')
        plt.show()

    def unzip(self, landmark):
        """
        将12个元组拆分成两个list，分别是x y 坐标
        :return: x, y list
        """
        x = []
        y = []
        for i in landmark:
            x.append(int(i[0]))
            y.append(int(i[1]))

        return x, y

    def draw_in_zeros(self):
        """
        使用这个方法将标注的坐标映射到指定大小的图片上面，最终返回一个zip后的list
        :return:
        """

    def parse_image_name(self, name):
        """
        解析图片的名称
        :param name:图片的名称
        :return:
        """
        name_name = name.split(';')[0]
        _ = name.split(';')[-1]
        name_format = _.split('-')[-1]
        landmark_list = _.split('-')[:-1]

        for idx, i in enumerate(landmark_list):
            landmark_list[idx] = (float(i.split(',')[0]), float(i.split(',')[1]))

        # 判断名称是否有问题
        if len(landmark_list) != 12:
            return False
        else:
            return {'name': name_name,
                    'landmark': landmark_list,
                    'format': name_format}


if __name__ == '__main__':
    LandMarkData().read_and_check()
