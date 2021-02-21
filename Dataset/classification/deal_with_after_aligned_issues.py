import os,sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageFilter
from PIL import Image
import cv2 as cv
import pandas as pd
from tqdm import tqdm
import traceback


class process_aligned_data:
    def __init__(self):
        self.img_dir = r'H:\after_correction'
        self.cls_csv = r'D:\plam_prog\Dataset\0214_aligned_cls_all.csv'
        self.save_path = r'H:\after_correction_rename'
        self.df = pd.read_csv(self.cls_csv)
        self.rename_for_cls(img_dir=self.img_dir)
    def rename_for_cls(self, img_dir):
        """
        得到gyq给的数据和查询到每个数据对应的cls 的csv之后就行如下操作步骤
        1. 遍历经过矫正后的数据
        2. 通过名称查询每一张图片所对应的类别
        3. 处理读取到的类别，然后rename it

        :return:
        """
        df = self.df
        img_list = os.listdir(img_dir)
        for idx, item in enumerate(tqdm(img_list)):
            try:
                img = Image.open(os.path.join(img_dir, item))

                # TODO 获取到名字
                name = item.split(';')[0] +'.' + item.split('.')[-1]

                # TODO using name to search cls
                try:
                    # print(name)
                    search_result = df.loc[df["ImgName"]==name]
                    cls = list(search_result['cls'])[0]
                    cls = cls.replace('[','').replace(']','').replace('\"','').replace('\'','').replace('(','').replace(')','')

                    cls_list = cls.split(',')
                    for i in range(len(cls_list)):
                        cls_list[i] = int(cls_list[i])
                    _ = []
                    for i in range(len(cls_list)//2):
                        _.append((cls_list[2*i],cls_list[2*i+1]))

                    self.save_src_and_gt(img, _, img_name=name)
                except Exception as e:
                    print(e)
                    continue

            except Exception as e:
                print(e)
                print('error')
                continue
    def save_src_and_gt(self, src, label, img_name):
        """
        这个方法使用来保存生成后的图片，分类标签的坐标保存在图片中
        如21130-女-28-右;1,2,2,1,3,4;
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
        for item in label:
            name_landmark += '%d,%d-' % (item[0], item[1])

        save_name = name_name + ';' + name_landmark + '.' + name_format
        try:
            src.save(os.path.join(self.save_path, save_name))
        except Exception as e:
            print(e, 'the image :%s error' % save_name)


        pass

if __name__ == '__main__':
    process_aligned_data()