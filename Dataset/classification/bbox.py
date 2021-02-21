"""
@author
time : 2021/2/7
"""

import os,sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class BBOX:
    def __init__(self):
        self.bbox_root= r'C:\Users\musk\Desktop\Palam_Data_AfterClear_xyxy(1)\Palam_Data_AfterClear_xyxy'
        self.gen_bbox_for_clear_data()
        pass
    def gen_bbox_for_clear_data(self):
        file_list = []
        for dirpath,dirname,filename in os.walk(self.bbox_root):
            for name in filename:
                file_list.append(os.path.join(dirpath,name))
                with open(os.path.join(dirpath,name)) as f:
                    data = f.read()
                    data = [float(i) for i in data.split(' ')]


                split_result = (os.path.join(dirpath, name).split('\\'))
                bbox_path = os.path.join(split_result[-3],split_result[-2])
                img_path = os.path.join('H:\Palam_Data_AfterClear',os.path.join(bbox_path,split_result[-1].replace('.txt', '.jpg')))

                try:
                    try:
                        img = Image.open(img_path)
                    except:
                        img = Image.open(img_path.replace('.jpg','.png'))

                    img = np.array(img,dtype='uint8')
                    x1 = img.shape[1] * data[0]
                    y1 = img.shape[0] * data[1]
                    x2 = img.shape[1] * data[2]
                    y2 = img.shape[0] * data[3]
                    plt.imshow(img)
                    print(x1,y1,x2,y2)
                    plt.gca().add_patch(plt.Rectangle(xy=(int(x1),int(y1)),
                                                      width=int(x2-x1),
                                                      height=int(y2-y1),
                                                      fill=False, linewidth=2))
                    # plt.plot((int(y1),int(x1)),(int(y2),int(x2)))
                    plt.show()
                except Exception as e :
                    print(e)
                    print('error')
        print(len(file_list))



if __name__ == '__main__':
    BBOX()