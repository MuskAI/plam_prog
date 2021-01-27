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

class LandMarkData:
    def __init__(self):
        self.save_path = 'H:/手掌关键点定位/after_resize'
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
            _image_and_landmark_list = self.read_csv(multi_landmark_path[i-1])
            _image_dir = multi_image_path[i-1]
            self.deal_with_all_image(_image_and_landmark_list, image_dir=_image_dir)

        print('The number of error image is : %d' % (len(self.error_list)))
        print(self.error_list)


    def deal_with_one_image(self, image_and_landmark_dict=None, image_dir=r'H:\手掌关键点定位\第二大批原始数据\第1批', size=192):
        """
        :param image_and_landmark_dict: 输入的是一个字典，只包含单张图片
        :param image_dir:输入的是图片所在的根目录，因为字典中只包含图片的名称
        :param size:希望将图片resize的大小
        :return:
        """
        try:
            # print(os.path.join(image_dir, image_and_landmark_dict['img_name']))
            src = Image.open(os.path.join(image_dir, image_and_landmark_dict['img_name']))
            if len(src.split()) ==3:
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

        resize_landmark = self.point_map(landmark=image_and_landmark_dict['landmark'], src_size=src.size,
                                         src_resize_size=(size,size))

        # self.landmark_after_resize(src, landmark=image_and_landmark_dict['landmark'], size=size)
        # resize to 192
        src_192 = src.resize((size, size))
        # self.visualize(src_192, resize_landmark)
        try:
            self.save_src_and_gt(src_192=src_192, landmark=resize_landmark, img_name=image_and_landmark_dict['img_name'])
        except Exception as e:
            print(e)
            return False
        return True
    # def landmark_after_resize(self, src, landmark, size):
    #     """
    #     获取resize后的坐标,
    #     :param src: 没有resize的原图 ，PIL Image
    #     :param landmark: 包含12个元组
    #     :return:
    #     """
    #
    #     # 缩小后的gt
    #     zeros = np.zeros_like(np.array(src.split()[0]))
    #     x, y = self.unzip(landmark)
    #
    #     # 把坐标点放在图中
    #     for i in range(12):
    #         zeros[x[i], y[i]] = 255
    #
    #     src_192 = src.resize((size, size))
    #     src_192 = np.array(src_192, dtype='uint8')
    #     # plt.imshow(src_192)
    #     # plt.show()
    #
    #     zeros_IMAGE = Image.fromarray(np.array(zeros, dtype='uint8'))
    #     zeros_IMAGE = zeros_IMAGE.resize((size, size))
    #     zeros_numpy = np.array(zeros_IMAGE)
    #     print('the max is %f' % (zeros_numpy.all()))
    #     # plt.imshow(zeros_numpy)
    #     # plt.show()
    #
    #     self.visualize(src_192, landmark=self.point_map(src_size=src.size, src_resize_size=(192, 192), landmark=landmark))
    #     print()
    #

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
            x[i] = int(x[i] / col_rate)
            y[i] = int(y[i] / row_rate)

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
            x = list(df.loc[[item+1, item+2, item+3, item+4, item+5, item+6, item+7, item+8, item+9, item+10, item+11, item+12]][0])
            y = list(df.loc[[item+1, item+2, item+3, item+4, item+5, item+6, item+7, item+8, item+9, item+10, item+11, item+12]][1])
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
            name_landmark += '%d,%d-' % (item[0],item[1])

        save_name = name_name + ';' + name_landmark + '.' +name_format
        print(save_name)
        try:
            src_192.save(os.path.join(self.save_path,save_name))
        except Exception as e:
            print(e, 'the image :%s error' % save_name)


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

if __name__ == '__main__':
    LandMarkData().multi_batch()
