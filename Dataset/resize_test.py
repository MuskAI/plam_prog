"""
created by haoran
time:2021/01/26
description:
1. 试探确定输入图片的尺寸大小
"""
from PIL import Image
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
class MyResize:
    def __init__(self):
        img = Image.open('./TEMPWORKSHOP/20001-女-27-右.jpg')
        img = img.resize((192, 192))
        img.save('./TEMPWORKSHOP/20001-女-27-右_192.jpg')
        pass

    def read_src(self):
        pass
    def read_gt(self):
        pass


    def resize_src_and_gt_check(self):
if __name__ == '__main__':
    MyResize()
