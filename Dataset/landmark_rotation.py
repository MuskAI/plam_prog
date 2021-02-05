"""
created by haoran
time : 2021/2/4
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, sys


class LandMarkRotation:
    def __init__(self):
        self.img_dir = r'H:\手掌关键点定位\after_resize_cleared\after_resize2'
        self.check()
        pass

    def check(self):
        img_list = os.listdir(self.img_dir)
        for idx, item in enumerate(img_list):
            img = Image.open(os.path.join(self.img_dir, item))
            angle = 90

            img = img.rotate(-angle, expand=True)
            parse_result = self.parse_image_name(item)
            landmark = parse_result['landmark']
            landmark = self.landmark_clockwise(landmark,angle)
            self.visualize(img, landmark)

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

        _landmark_list = []
        for idx, item in enumerate(landmark_list):
            _landmark_list.append(item[0])
            _landmark_list.append(item[1])
        landmark_list = _landmark_list

        return {'name': name_name,
                'landmark': landmark_list,
                'format': name_format}

    def visualize(self, img, landmark):
        """
        :param img: PIL Image 类型
        :param landmark: 一个list 其中每个点都是一个元组
        :return: 直接可视化结果
        """
        print('now we are in visualize')
        # print(landmark.shape)

        # print(landmark.shape)
        # print(landmark)
        img = np.array(img, dtype='uint8')
        plt.figure('visualize')
        plt.imshow(img)
        y = []
        x = []
        for i in range(12):
            x.append(int(landmark[2 * i]))
            y.append(int(landmark[2 * i + 1]))
        print(x)
        print(y)
        print()
        plt.plot(y, x, '*')
        plt.show()

    def landmark_clockwise(self, landmark, angle):
        """
        x1=xcos(β)-ysin(β);
        y1=ycos(β)+xsin(β);
        :param landmark:
        :return:
        """
        print(angle)
        for i in range(12):
            x = landmark[2 * i]
            y = landmark[2 * i + 1]
            print(x, y)

            y1 = y * round(np.cos(np.deg2rad(angle)), 2) + x * round(np.sin(np.deg2rad(angle)), 2)
            landmark[2 * i] = x * round(np.cos(np.deg2rad(angle)), 2) - y * round(np.sin(np.deg2rad(angle)), 2)
            landmark[2 * i + 1] = -y1 + 192

        return landmark


if __name__ == '__main__':
    LandMarkRotation()
