"""
created by haoran
time : 2021/2/4
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import random
import traceback

class LandMarkRotation:
    def __init__(self):
        self.img_dir = r'H:\手掌关键点定位\after_resize_cleared\after_resize2'
        self.save_path = r'H:\手掌关键点定位\argument_data'
        # self.check()
        pass

    def check(self):
        img_list = os.listdir(self.img_dir)
        for idx, item in enumerate(img_list):
            img_ = Image.open(os.path.join(self.img_dir, item))
            angle = [90,180,270]
            angle = random.sample(angle,1)[0]
            random_num = np.random.randint(-5, 5,1)[0]
            angle += random_num

            img = img_.rotate(-angle)
            parse_result = self.parse_image_name(item)
            landmark = parse_result['landmark']
            print('landmark:',landmark)
            self.visualize(img_, landmark)
            landmark = self.landmark_clockwise(landmark, angle)
            print('landmark:', landmark)
            self.visualize(img, landmark)
    def using_when_training(self,name):
        img_ = Image.open(os.path.join(self.img_dir, name))
        angle = [90, 180, 270]
        angle = random.sample(angle, 1)[0]
        random_num = np.random.randint(-5, 5, 1)[0]
        angle += random_num

        img = img_.rotate(-angle)
        parse_result = self.parse_image_name(name)
        landmark = parse_result['landmark']
        landmark = self.landmark_clockwise(landmark, angle)
        return img,landmark
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

        _landmark = []
        center_point = [96, 96]
        for i in range(12):
            y = landmark[2 * i]
            x = landmark[2 * i + 1]

            x1 = (x - center_point[0]) * round(np.cos(np.deg2rad(angle)), 2) - \
                 (y-center_point[1]) * round(np.sin(np.deg2rad(angle)), 2) + center_point[0]
            y1 = (x - center_point[0]) * round(np.sin(np.deg2rad(angle)), 2) + \
                 (y-center_point[1]) * round(np.cos(np.deg2rad(angle)), 2) + center_point[1]
            _landmark.append(y1)
            _landmark.append(x1)

        return _landmark

    def gen_argument_dataset(self):
        img_list = os.listdir(self.img_dir)
        for idx, item in enumerate(img_list):
            img_ = Image.open(os.path.join(self.img_dir, item))
            angle = [90,180,270]
            angle = random.sample(angle,1)[0]
            random_num = np.random.randint(-5, 5,1)[0]
            angle += random_num

            img = img_.rotate(-angle)
            parse_result = self.parse_image_name(item)
            landmark = parse_result['landmark']
            print('landmark:',landmark)
            landmark = self.landmark_clockwise(landmark, angle)
            print('landmark:', landmark)

            # TODO convert landmark
            _landmark = []
            for i in range(12):
                _landmark.append((landmark[2*i],landmark[2*i+1]))
            self.save_src_and_gt(img,_landmark,parse_result['name']+'.'+parse_result['format'])
if __name__ == '__main__':
    LandMarkRotation().gen_argument_dataset()
