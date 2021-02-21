"""
@author  HAORAN
time: 2021/2/4
"""
import hashlib
import os, sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import traceback
from multiprocessing import Pool
from rich.progress import track
import base64
import time


class GenLandmarkCode:
    """
    对于有landmark的数据
    """

    def __init__(self):
        self.landmark_data_dir = r'H:\手掌关键点定位\after_resize_cleared\after_resize2'
        self.error_list = []
        self.img_name = []
        self.img_code = []
        self.img_landmark = []
        pass

    def gen_img_coding(self, path):
        """
                生成图片的编码，(密码本)

        :param item: 图片路径
        :return:
        """
        with open(os.path.join(self.landmark_data_dir, path), "rb") as imageFile:
            img_str = base64.b64encode(imageFile.read())
        code = hashlib.md5(img_str).hexdigest()
        return code

    def multi_batch(self):
        """
        每个批次的数据，逐一处理，读取CSV在这里
        :return:
        """

        img_name = []
        img_landmark = []
        code_list = []
        for i in range(1, 10):
            print('In multi_batch process: %d / 9' % (i))
            _image_and_landmark_list = self.read_csv('./landmark_gt/第%d批check.csv' % (i))
            length = len(_image_and_landmark_list)
            for idx, item in enumerate(_image_and_landmark_list):
                print('\r', idx, '/', length, end='')
                try:
                    code = self.gen_img_coding(os.path.join('H:/手掌关键点定位/第二大批原始数据/第%d批' % (i), item['img_name']))
                    code_list.append(code)
                    img_name.append(item['img_name'])
                    img_landmark.append(item['landmark'])
                except Exception as e:
                    print(e)
                    self.error_list.append(item)
        self.img_code = code_list
        self.img_name = img_name
        self.img_landmark = img_landmark
        data = {'ImgName': img_name, 'ImgMD5': code_list, 'landmark': img_landmark}
        df = pd.DataFrame(data)
        df.to_csv('landmark_md5.csv')
        print('The number of error image is : %d' % (len(self.error_list)))
        print(self.error_list)

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


class GenClsCode:
    """
    对于无landmark的数据
    """

    def __init__(self):
        self.rootdir = r'H:\Palam_Data_AfterClear'  # 指明被遍历的文件夹
        self.parentCls = ['1肠胃', '3妇科', '4肝胆系统疾病', '5男科', '7神经系统', '8五官', '9心脑血管疾病']
        self.error_list = []
        pass

    def gen_img_coding(self, path):
        """
                生成图片的编码，(密码本)

        :param item: 图片路径
        :return:
        """
        with open(path, "rb") as imageFile:
            img_str = base64.b64encode(imageFile.read())
        code = hashlib.md5(img_str).hexdigest()
        return code

    def class_encode(self):
        code_list = []
        name_list = []
        for idx1, pcls in enumerate(self.parentCls):
            # 大类文件夹
            parent_path = os.path.join(self.rootdir, pcls)
            chirld_path = os.listdir(parent_path)
            for idx2, item in enumerate(chirld_path):
                # 小类文件夹
                img_dir = os.path.join(parent_path, item)
                print("Now we are in ", item, "(", idx2, '/', len(chirld_path), '):')
                for idx3, temp_image_name in enumerate(os.listdir(img_dir)):
                    # 遍历每张图片
                    print(temp_image_name, ' :(', idx3, '/', len(os.listdir(img_dir)), ')')
                    try:
                        code = self.gen_img_coding(os.path.join(img_dir, temp_image_name))
                        code_list.append(code)
                        name_list.append(temp_image_name)
                    except Exception as e:
                        print(e)
                        self.error_list.append(os.path.join(img_dir, temp_image_name))
        data = {'ImgName': name_list, 'ImgMD5': code_list}
        df = pd.DataFrame(data)
        df.to_csv('cls_md5.csv')


class GenClsDataset():
    """
    生成数据
    """

    def __init__(self):

        self.df = pd.read_csv('cls_md5.csv')
        self.aligned_dir = r'H:\after_correction'
        self.error_list = []
        self.full_label_for_img()
        pass

    def read_csv(self, csv_path):
        """
        遍历每一个含有landmark的数据，为每一个数据增加标签信息
        :return:
        """
        name_list = []
        code_list = []
        landmark_list = []
        same_name_list = []
        cls_list = []
        df = pd.read_csv(csv_path)
        length = df.shape[0]

        for idx,row in df.iterrows():
            print(idx, '/', length//4)

            name = row[1]
            code = row[2]
            landmark = row[3]
            same_name, cls = self.search_same(target_code=code)
            name_list.append(name)
            code_list.append(code)
            landmark_list.append(landmark)
            same_name_list.append(same_name)
            cls_list.append(cls)

        data = {'ImgName': name_list, 'ImgMD5': code_list,
                'landmark': landmark_list,'same_name':same_name_list,'cls':cls_list}
        df = pd.DataFrame(data)
        df.to_csv('landmark_md5_cls.csv')
    def search_same(self, target_code):

        df = self.df
        cls_list = []

        result = df.loc[df['ImgMD5']==target_code]

        find_name=list(result['ImgName'])
        for name in find_name:
            cls_list.append(str((name.split('_')[1], name.split('_')[2])))
        #
        # for idx,row in df.iterrows():
        #
        #     name = row[1]
        #     search_code = row[2]
        #     if search_code == target_code:
        #         find_name.append(name)
        #         cls_list.append(str((name.split('_')[1], name.split('_')[2])))

        if len(find_name) == 0:
            return '', ''
        else:
            return str(find_name), str(cls_list)

    def parse_landmark(self, landmark):
        landmark = landmark.replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace("\'", '')
        landmark = landmark.split(',')

        landmark_out = []
        for i in range(12):
            # print(landmark[2*i])
            # print(landmark[2 * i + 1])
            landmark_out.append(int(landmark[2 * i]))
            landmark_out.append(int(landmark[2 * i + 1]))
        return landmark_out

    def full_label_for_img(self):
        """
        拿到gyq给的矫正后的图片，为每个图片补充标签信息
        1. 遍历每一张图片，parse每一张图片的name
        2. 根据每一张图片唯一的name，读取cls_md5.csv 获取md5 code
        3. 根据获取的md5 code 去search 相同label的图片
        :return:
        """
        name_list = []
        code_list = []
        landmark_list = []
        same_name_list = []
        cls_list=[]
        df = self.df
        aligned_img_list = os.listdir(self.aligned_dir)
        for idx, item in enumerate(tqdm(aligned_img_list)):
            start = time.time()

            end1 = time.time()
            try:
                name = item.split(';')[0]
                format = item.split('.')[-1]
                name = name + '.' + format

                # TODO 查找name对应的md5
                # print(name)
                search_result = df.loc[df['ImgName'] == name]
                code = list(search_result['ImgMD5'])[0]
                end2 = time.time()
                same_name, cls = self.search_same(target_code=code)
                # print(same_name)

                name_list.append(name)
                code_list.append(code)
                same_name_list.append(same_name)
                cls_list.append(cls)
                end3 = time.time()


            except Exception as e:
                self.error_list.append(name)
        data = {'ImgName': name_list, 'ImgMD5': code_list,
                'same_name':same_name_list,'cls':cls_list}

        df = pd.DataFrame(data)
        df.to_csv('0214_aligned_cls_all.csv')
        print(print("\033[4;31;43mThe Error list is :\033[0m"))
        print(len(self.error_list))
        print(self.error_list)


if __name__ == '__main__':


    GenClsDataset()
    # GenLandmarkCode().multi_batch()
    # GenClsCode().class_encode()
