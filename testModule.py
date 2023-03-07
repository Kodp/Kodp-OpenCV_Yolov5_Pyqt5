# 测试模块
import findContourAngle
import cv2 as cv
import numpy as np
import os
import os.path as osp
import math
import operator
from utilsMy import *
from functools import reduce
import torch
from main import init

# model = init()

# single_paths = ['E:/Grapro/yolo_trepro/pic\cut_num0001',
#                 'E:/Grapro/yolo_trepro/pic\cut_num0002']
#
# for path in single_paths:
#     imgs , nums = [], []  # nums = imgs = [] imgs 和 nums是一个东西
#     for i in os.listdir(path):
#         img = cv.imread(os.path.join(path,i))
#         imgs.append(img)
#     res = model(imgs, size=416).xywh
#     for i in range(len(res)):
#         c = res[i][res[i][:, 0].argsort()]  # 按第0列排序 https://codeantenna.com/a/91tbTDDuvl
#         if len(c) == 0:  # 没有识别到数字
#             nums.append(-1)
#             continue
#         num = 0
#         for j in range(len(c)):
#             num = num * 10 + int(c[j][5])
#         nums.append(num)
#     print(nums)


print(osp.splitext(osp.split(r'E:\Grapro\yolo_trepro\src\test\cut\a.jpg')[1]))

