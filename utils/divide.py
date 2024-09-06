import os
import random
import glob
from pathlib import Path, PurePath
import shutil
from tqdm import tqdm

TRAIN_DIR = 'train'
VAL_DIR = 'val'
TEST_DIR = 'test'
DIVIDE_DIR = [TRAIN_DIR, VAL_DIR, TEST_DIR]


def classify_divide(dataset_path, train_ratio=0.7, val_ratio=0.2, seed=None):
    '''
    用于分类任务的数据集划分器
    root/
    |-- class1/
    |-- class2/

    to:
    root/
    |-- train/
    |   |-- class1/
    |   |-- class2/
    |-- val/
    |   |-- class1/
    |   |-- class2/
    |-- test/
    |   |-- class1/
    |   |-- class2/
    '''

    imgs_list = glob.glob(f'{dataset_path}/**/*.jpg', recursive=True)
    cls_list = os.listdir(dataset_path)

    if seed:
        random.seed(seed)

    num = len(imgs_list)
    trainval_num = int(num * (train_ratio + val_ratio))  # 训练+校验集数量
    train_num = int(trainval_num * train_ratio / (train_ratio + val_ratio))  # 训练集数量

    trainval_list = random.sample(imgs_list, trainval_num)  # 随机选取trainval_num数量的图片
    train_list = random.sample(trainval_list, train_num)  # 从trainval中随机选取train_num数量的图片

    # 创建文件夹
    for cls in cls_list:
        for dir in ['train', 'val', 'test']:
            new_dir = os.path.join(dataset_path, dir, cls)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

    # 移动图片到对应目录
    for img_path in tqdm(imgs_list):
        if img_path in train_list:
            target_dir = 'train'
        elif img_path in trainval_list:
            target_dir = 'val'
        else:
            target_dir = 'test'
        target_path = os.path.join(dataset_path, target_dir, os.path.relpath(img_path, dataset_path))
        shutil.move(img_path, target_path)

    # 删除原类别文件夹
    for cls in cls_list:
        os.rmdir(os.path.join(dataset_path, cls))

def create_divide_dir():
    '''
    用于创建数据集划分子文件夹
    '''
    pass

def remove_old_dir():
    '''
    用于删除旧数据集类文件夹
    '''
    pass


if __name__ == "__main__":
    classify_divide(r'D:\Projects\python\run_red_light_detect\train\datasets\traffic_light2', seed=0)
