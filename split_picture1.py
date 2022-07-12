

import numpy as np
import os
import shutil



file_path = "/home/lixulong/Desktop/untitled folder/single_image/0516_3/picture"   #更换你数据集的文件名
# file_path = "/home/lixulong/Desktop/untitled folder/single_image/0516_3/untitled folder/ceshi"
path_list = os.listdir(file_path)                 #会历遍文件夹内的文件并返回一个列表
path_name = []

for i in path_list:
    path_name.append(file_path + "/" + i)
# 排序一下
path_name.sort()

train_path = []
test_path = []
trains_idx = []
tests_idx = []
for i in range(1):             #图片的类别数为21
    # start = i * 35             #每一类图片数为21
    # end = (i + 1) * 35
    idx = np.arange(0, 34)
    np.random.shuffle(idx)
    train_idx = idx[0:27]       #训练集每一类选取随机排序后  0-80作为训练集
    test_idx = idx[27:]         #测试集每一类选取随机排序后  80-100作为测试集合
    trains_idx.extend(train_idx)
    tests_idx.extend(test_idx)
path_name = np.array(path_name)
train_path = path_name[trains_idx]
test_path = path_name[tests_idx]

for file_name in train_path:
    shutil.copy(file_name, "/home/lixulong/Desktop/untitled folder/single_image/0516_3/train")  #你提前建立好的存放训练集文件的文件夹
for file_name in test_path:
    shutil.copy(file_name, "/home/lixulong/Desktop/untitled folder/single_image/0516_3/test")   #你提前建立好的存放测试集文件的文件夹
