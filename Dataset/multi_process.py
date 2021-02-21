"""
@author  HAORAN
time: 2021/2/4
多进程生成
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
        find_name = []
        length = df.shape[0]
        cls_list = []
        for idx,row in df.iterrows():

            name = row[1]
            search_code = row[2]
            if search_code == target_code:
                find_name.append(name)
                cls_list.append(str((name.split('_')[1], name.split('_')[2])))
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
            if item.split('_')[1] == '10':
                continue
            try:
                name = item.split(';')[0]
                format = item.split('.')[-1]
                name = name + '.' + format

                # TODO 查找name对应的md5
                # print(name)
                search_result = df.loc[df['ImgName'] == name]
                code = list(search_result['ImgMD5'])[0]
                same_name, cls = self.search_same(target_code=code)
                name_list.append(name)
                code_list.append(code)
                same_name_list.append(same_name)
                cls_list.append(cls)



            except Exception as e:
                self.error_list.append(name)
        data = {'ImgName': name_list, 'ImgMD5': code_list,
                'landmark': landmark_list,'same_name':same_name_list,'cls':cls_list}
        df = pd.DataFrame(data)
        df.to_csv('0214_aligned_cls.csv')
        print(print("\033[4;31;43mThe Error list is :\033[0m"))
        print(len(self.error_list))
        print(self.error_list)




def full_label_for_img():
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
        if item.split('_')[1] == '10':
            continue
        try:
            name = item.split(';')[0]
            format = item.split('.')[-1]
            name = name + '.' + format

            # TODO 查找name对应的md5
            # print(name)
            search_result = df.loc[df['ImgName'] == name]
            code = list(search_result['ImgMD5'])[0]
            same_name, cls = self.search_same(target_code=code)
            name_list.append(name)
            code_list.append(code)
            same_name_list.append(same_name)
            cls_list.append(cls)



        except Exception as e:
            self.error_list.append(name)
    data = {'ImgName': name_list, 'ImgMD5': code_list,
            'landmark': landmark_list,'same_name':same_name_list,'cls':cls_list}
    df = pd.DataFrame(data)
    df.to_csv('0214_aligned_cls.csv')
    print(print("\033[4;31;43mThe Error list is :\033[0m"))
    print(len(self.error_list))
    print(self.error_list)

if __name__ == '__main__':
