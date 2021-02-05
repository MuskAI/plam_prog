import csv
import hashlib
import os
import re
import os.path
import time
import base64

start_time = time.time()

rootdir = r'E:\Palam_Data'  # 指明被遍历的文件夹
parentCls = ['1肠胃', '3妇科', '4肝胆系统疾病', '5男科', '7神经系统', '8五官', '9心脑血管疾病']

class GenClsCode:
    def __init__(self):
        self.rootdir = r'E:\Palam_Data'  # 指明被遍历的文件夹
        self.parentCls = ['1肠胃', '3妇科', '4肝胆系统疾病', '5男科', '7神经系统', '8五官', '9心脑血管疾病']

        pass


    def class_encode(self):
        for idx1,pcls in enumerate(parentCls):
            # 大类文件夹
            parent_path = os.path.join(rootdir,pcls)
            chirld_path = os.listdir(parent_path)
            for idx2,item in enumerate(chirld_path):
                # 小类文件夹
                img_dir = os.path.join(parent_path, item)
                print("Now we are in ", item, "(", idx2, '/', len(chirld_path), '):')
                for idx3, temp_image_name in enumerate(os.listdir(img_dir)):
                    # 遍历每张图片
                    print(temp_image_name, ' :(', idx3, '/', len(os.listdir(img_dir)), ')')
                    try:

import csv

print("开始output")
f = open('Image_Md5_Code_ALL_2.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['Name', 'MD5Code'])
for (name, code) in zip(OutputPathName, OutputCode):
    csv_writer.writerow([name, code])
f.close()
end_time = time.time()

print("total time is :", end_time - start_time)
