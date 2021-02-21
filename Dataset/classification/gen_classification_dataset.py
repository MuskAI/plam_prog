"""
@author haoran
time:2021/2/7
"""
import os,sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageFilter
from PIL import Image
import cv2 as cv
import pandas as pd
from tqdm import tqdm

import base64
import hashlib
import traceback
class GenClsDataset:
    """
    使用这个类去生成没有矫正的分类数据
    大概的步骤是：
    1. 读取bbox，进行切分
    2. 将每一个切分后的图保存到指定位置
    3. 每一张图片的gt保存在切分图的名字中name;landmar;(1,1),(1,2)
    """
    def __init__(self):
        self.bbox_root = r'C:\Users\musk\Desktop\Palam_Data_AfterClear_xyxy(1)\Palam_Data_AfterClear_xyxy'
        self.save_path = r'H:\分类数据'
        self.df = pd.read_csv('../cls_md5.csv')
        self.error_list = []
        self.save_path_for_landmark = r'H:\裁剪好准备输入定位网络的数据'
        self.crop_data()
        # self.gen_data_for_landmark_detection()



    def crop_data(self):
        file_list = []
        name_list = []
        code_list = []
        same_name_list = []
        cls_list = []
        for dirpath,dirname,filename in tqdm(os.walk(self.bbox_root)):
            for name in tqdm(filename):
                file_list.append(os.path.join(dirpath,name))
                try:
                    with open(os.path.join(dirpath,name)) as f:
                        data = f.read()
                        data = [float(i) for i in data.split(' ')]
                except:
                    self.error_list.append(os.path.join(dirpath,name))
                    continue

                split_result = (os.path.join(dirpath, name).split('\\'))
                bbox_path = os.path.join(split_result[-3],split_result[-2])
                img_path = os.path.join('H:\Palam_Data_AfterClear',os.path.join(bbox_path,split_result[-1].replace('.txt', '.jpg')))

                try:
                    try:
                        img = Image.open(img_path)
                    except:
                        img_path = img_path.replace('.jpg','.png')
                        img = Image.open(img_path)

                    img = np.array(img,dtype='uint8')
                    x1 = int(img.shape[1] * data[0])
                    y1 = int(img.shape[0] * data[1])
                    x2 = int(img.shape[1] * data[2])
                    y2 = int(img.shape[0] * data[3])

                    img = img[y1:y2,x1:x2,:]
                    img = Image.fromarray(img)
                    img = img.resize((512, 512))


                    # TODO search code
                    try:
                        name = img_path.split('\\')[-1]
                        code = self.gen_img_coding(img_path)
                        same_name, cls = self.search_same(target_code=code)
                        # print(same_name)
                        # print(cls)
                        self.save_src_and_gt(img, cls, img_name=name)
                    except Exception as e:
                        print(e)

                except Exception as e :
                    print(e)
                    print('error')
                    continue
                name_list.append(name)
                code_list.append(code)
                same_name_list.append(same_name)
                cls_list.append(cls)

        data = {'ImgName': name_list, 'ImgMD5':code_list,
                'same_name':same_name_list,'cls':cls_list}
        df = pd.DataFrame(data)
        df.to_csv('all_data_label.csv')
        print(self.error_list)

    def gen_data_for_landmark_detection(self,pad=100):
        file_list = []
        name_list = []
        code_list = []
        same_name_list = []
        cls_list = []
        for dirpath,dirname,filename in tqdm(os.walk(self.bbox_root)):
            for name in tqdm(filename):
                file_list.append(os.path.join(dirpath,name))
                try:
                    with open(os.path.join(dirpath,name)) as f:
                        data = f.read()
                        data = [float(i) for i in data.split(' ')]
                except:
                    self.error_list.append(os.path.join(dirpath,name))
                    continue

                split_result = (os.path.join(dirpath, name).split('\\'))
                bbox_path = os.path.join(split_result[-3],split_result[-2])
                img_path = os.path.join('H:\Palam_Data_AfterClear',os.path.join(bbox_path,split_result[-1].replace('.txt', '.jpg')))

                try:
                    try:
                        img = Image.open(img_path)
                    except:
                        img_path = img_path.replace('.jpg','.png')
                        img = Image.open(img_path)

                    img = np.array(img,dtype='uint8')
                    x1 = int(img.shape[1] * data[0])
                    y1 = int(img.shape[0] * data[1])
                    x2 = int(img.shape[1] * data[2])
                    y2 = int(img.shape[0] * data[3])


                    img = img[y1-pad:y2+pad,x1-pad:x2+pad,:]
                    img = Image.fromarray(img)
                    img = img.resize((192, 192))

                    # plt.imshow(img)
                    # plt.show()
                    # print((x1,y1,x2,y2))
                    self.save_src_for_landmark(src=img, name=img_path.split('\\')[-1], bbox=(x1,y1,x2,y2))

                except Exception as e :
                    print(e)
                    print('error')
                    continue

        data = {'ImgName': name_list, 'ImgMD5':code_list,
                'same_name':same_name_list,'cls':cls_list}
        df = pd.DataFrame(data)
        df.to_csv('all_data_label.csv')
        print(self.error_list)

    def save_src_for_landmark(self,src,name,bbox):
        # 首先检查保存的路径
        if not os.path.exists(
                self.save_path_for_landmark):
            traceback.print_exc('please enter legal save_path')

        # 生成保存的文件名
        name_name = name.split('.')[0]
        name_format = name.split('.')[-1]
        # print(name_name)
        # print(bbox)
        save_name = name_name + ';' + '%d,%d,%d,%d' % bbox + '.' + name_format
        # print(save_name)
        try:
            src.save(os.path.join(self.save_path_for_landmark, save_name))
        except Exception as e:
            print(e, 'the image :%s error' % save_name)

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
            name_landmark += '%.2f,%.2f-' % (item[0], item[1])

        save_name = name_name + ';' + name_landmark + '.' + name_format
        try:
            src.save(os.path.join(self.save_path, save_name))
        except Exception as e:
            print(e, 'the image :%s error' % save_name)


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

    def readable_check(self):
        pass
    def search_same(self, target_code):

        df = self.df
        find_name = []
        length = df.shape[0]
        cls_list = []

        result = df.loc[df['ImgMD5']==target_code]

        find_name=list(result['ImgName'])
        for name in find_name:
            cls_list.append((int(name.split('_')[1]), int(name.split('_')[2])))


        return find_name, cls_list
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




if __name__ == '__main__':
    GenClsDataset()