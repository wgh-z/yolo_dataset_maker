# 数据处理
import os
# import cv2
import random
import yaml
# import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from functools import partial

from utils.format_io import xml2txt, txt2xml
from utils.tools import get_image_info


class DatasetFormat:
    def __init__(
            self,
            dataset_path,
            image_dir='images',
            background_dir='backgrounds',
            xml_label_dir='annotations',
            txt_label_dir='labels',
            ):
        self.dataset_path = dataset_path
        self.image_dir = image_dir
        self.bg_dir = background_dir
        self.xml_dir = xml_label_dir
        self.txt_dir = txt_label_dir

        self.image_path = os.path.join(self.dataset_path, self.image_dir)
        self.bg_path = os.path.join(self.dataset_path, self.bg_dir)
        self.xml_path = os.path.join(self.dataset_path, self.xml_dir)
        self.txt_path = os.path.join(self.dataset_path, self.txt_dir)
        self.dataset_name = os.path.split(self.dataset_path)[-1]  # 获取数据集名

    def data_divide(self, train_ratio=0.7, val_ratio=0.2, seed=None):
        """数据集划分"""

        print('开始划分数据集。。。')
        trainval_ratio = train_ratio + val_ratio
        print(f'训练集：{train_ratio}，校验集：{val_ratio}，测试集：{1-trainval_ratio}')

        imgs = [f'./{self.image_dir}/{img}\n' for img in os.listdir(self.image_path)]
        bg_imgs = [f'./{self.bg_dir}/{img}\n' for img in os.listdir(self.bg_path)]

        if seed is not None:
            random.seed(seed)
        random.shuffle(imgs)
        random.shuffle(bg_imgs)

        num = len(imgs)
        num_bg = len(bg_imgs)

        train_img_size = round(num * train_ratio)  # 训练集数量
        trainval_img_size = round(num * trainval_ratio)
        train_imgs = imgs[:train_img_size]
        val_imgs = imgs[train_img_size:trainval_img_size]
        test_imgs = imgs[trainval_img_size:]
        train_bg_img_size = round(num_bg * train_ratio)
        trainval_bg_img_size = round(num_bg * trainval_ratio)
        train_bg_imgs = bg_imgs[:train_bg_img_size]
        val_bg_imgs = bg_imgs[train_bg_img_size:trainval_bg_img_size]
        test_bg_imgs = bg_imgs[trainval_bg_img_size:]

        train = train_imgs + train_bg_imgs
        val = val_imgs + val_bg_imgs
        test = test_imgs + test_bg_imgs
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        datasets = {
            'train': train,
            'val': val,
            'test': test,
            }

        for dataset_name, imgs in datasets.items():
            with open(os.path.join(self.dataset_path, f'{dataset_name}.txt'), 'w') as f:
                for img in imgs:
                    f.write(img)

    def voc2yolo(self, classes):
        """voc数据集转yolo格式"""

        print('开始voc转yolo。。。')
        with open(f'{self.dataset_path}/classes.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(classes))  # 保存为yolo格式类别文件

        os.makedirs(self.txt_path, exist_ok=True)
        xml_to_txt = partial(
            xml2txt,
            xml_label_path=self.xml_path,
            txt_label_path=self.txt_path,
            classes=classes
            )

        imgs = os.listdir(self.image_path)
        for img in tqdm(imgs):
            img_id = os.path.splitext(img)[0]
            xml_to_txt(img_id)  # 生成labels文件夹下的txt

    def yolo2voc(self, classes):
        """yolo数据集转voc格式"""
        print('开始yolo转voc。。。')

        os.makedirs(self.xml_path, exist_ok=True)
        txt_to_xml = partial(
            txt2xml,
            xml_label_path=self.xml_path,
            txt_label_path=self.txt_path,
            classes=classes
            )

        imgs = os.listdir(self.image_path)
        for img in tqdm(imgs):
            img_id = os.path.splitext(img)[0]
            img_w, img_h, img_c = get_image_info(os.path.join(self.image_path, img))
            txt_to_xml(img_id, img_w, img_h, img_c)

    def get_labels(self, annotation_type='voc'):
        """获取数据集中的类别和类别数"""
        assert annotation_type in ['voc', 'yolo'], 'annotation_type must be voc or yolo'

        if annotation_type == 'voc':
            classes = []
            xmls = os.listdir(self.xml_path)  # 文件夹下所有文件的文件名列表
            for xml_name in xmls:
                tree = ET.parse(os.path.join(self.xml_path, xml_name))
                root = tree.getroot()
                # xml文件中每个object标签的格式为：<object><name>类别名</name></object>
                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    if cls not in classes:
                        classes.append(cls)

        elif annotation_type == 'yolo':
            with open(os.path.join(self.dataset_path, 'classes.txt'), 'r', encoding='utf-8') as f:
                classes = f.readlines()
            classes = [cls.strip('\n') for cls in classes]

        return classes

    # 批量修改VOC数据集中xml的标签名称
    def change_xml_label(self, old_class, new_class):
        file_names = os.listdir(self.xml_path)
        count = 0
        for file in file_names:
            if file.endswith('xml'):
                file = os.path.join(self.xml_path, file)
                tree = ET.parse(file)
                root = tree.getroot()
                for obj in root.iter('object'):
                    cls = obj.find('name')  # 一个object节点下面有一个name节点
                    if cls.text == old_class:  # 修改前的名称
                        cls.text = new_class  # 修改后的名称
                        count += 1
                tree.write(file, encoding='utf-8')  # 写进原始的xml文件并避免原始xml中文字符乱码
        print("替换了%d次"%count)

    # 批量删除VOC数据集中xml的标签名称
    def delete_xml_label(self, delete_class:list):
        annotations_path = os.path.join(self.dataset_path, 'annotations')
        file_names = os.listdir(annotations_path)
        count = 0
        for file in file_names:
            if file.endswith('xml'):
                file = os.path.join(annotations_path, file)
                tree = ET.parse(file)
                root = tree.getroot()
                for obj in root.findall('object'):  # findall返回的是一个列表
                    cls = obj.find('name')  # 一个object节点下面有一个name节点
                    if cls.text in delete_class:
                        root.remove(obj)
                        count += 1
                tree.write(file, encoding='utf-8')  # 写进原始的xml文件并避免原始xml中文字符乱码
        print("删除了%d次"%count)

    def create_yaml(self, classes):
        """生成数据集描述yaml"""

        args = [
                {'path': self.dataset_name},
                {'train':'train.txt'},
                {'val':'val.txt'},
                {'test':'test.txt'}
        ]
        names = {'names':{}}
        for i, class_name in enumerate(classes):
            names['names'][i] = class_name

        yaml_name = os.path.join(self.dataset_path, f'{self.dataset_name}.yaml')
        with open(yaml_name, 'w', encoding='utf-8') as f:
            f.write('# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]\n')
            for arg in args:
                yaml.safe_dump(arg, f, default_flow_style=False)
            f.write('\n# Classes\n')
            yaml.safe_dump(names, f, default_flow_style=False)

    def __call__(self, train_ratio=0.7, val_ratio=0.2, seed=None, annotation_type='voc'):
        """数据格式化"""

        classes = self.get_labels(annotation_type=annotation_type)
        if annotation_type == 'voc':
            self.voc2yolo(classes)
        elif annotation_type == 'yolo':
            self.yolo2voc(classes)

        self.data_divide(train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        self.create_yaml(classes)
        print(f'总共有{len(classes)}个类别：{classes}')


if __name__ == '__main__':
    import time
    dataset_path = 'datasets/detention'

    data_process = DatasetFormat(dataset_path)
    start = time.time()
    data_process(
        train_ratio=0.7,
        val_ratio=0.2,
        seed=0,
        annotation_type='voc'
        )
    end = time.time()
    print(f'耗时{end-start}秒')
