# 数据处理
import os
import random
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET

from utils.format_io import xyxy2xywhn


class Data_process:
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
        self.image_path = os.path.join(self.dataset_path, image_dir)
        self.bg_path = os.path.join(self.dataset_path, background_dir)
        self.xml_label_path = os.path.join(self.dataset_path, xml_label_dir)
        self.txt_label_path = os.path.join(self.dataset_path, txt_label_dir)
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

    def xml2txt(self, img_id, classes:list):
        try:
            tree = ET.parse(f'{self.xml_label_path}/{img_id}.xml')  # 解析xml文件
            root = tree.getroot()  # 获得对应的键值对
            size = root.find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)

            labels = ''
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                # 如果类别不是对应在我们预定好的class文件中，或difficult==1则跳过
                if cls not in classes or int(difficult) == 1:
                    continue
                # 通过类别名称找到id
                class_id = classes.index(cls)

                xml_box = obj.find('bndbox')
                xyxy = [float(xml_box.find(key).text) for key in ['xmin', 'ymin', 'xmax', 'ymax']]
                xywhn = xyxy2xywhn(xyxy, img_w, img_h)  # b_box转为(x,y,w,h)，并归一化
                labels += f'{class_id} {" ".join([str(coord) for coord in xywhn])}\n'

        except FileNotFoundError as e:  # 背景图
            with open(f'{self.txt_label_path}/{img_id}.txt', 'w', encoding='utf-8') as txt_file:
                return
        except AttributeError as e:  # 忽略错误标注
            return
        else:
            with open(f'{self.txt_label_path}/{img_id}.txt', 'w', encoding='utf-8') as txt_file:
                txt_file.write(labels)  # class_id x y w h

    def txt2xml(self, img_name, img_w, img_h, img_d, file_path, labeldicts):
        # 创建Annotation根节点
        root = ET.Element('annotation')
        root.text = '\n\t'
        root.tail = '\n'
        # 创建folder子节点
        folder = ET.SubElement(root, 'folder')
        folder.text = 'images'
        folder.tail = '\n\t'
        # 创建filename子节点，无扩展名
        filename = ET.SubElement(root, 'filename')
        filename.text = str(img_name)
        filename.tail = '\n\t'
        # 创建path子节点
        path = ET.SubElement(root, 'path')
        path.text = str(file_path)
        path.tail = '\n\t'
        # 创建source子节点
        source = ET.SubElement(root, 'source')
        source.text = '\n\t\t'
        source.tail = '\n\t'
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'
        database.tail = '\n\t'

        # 创建size子节点
        sizes = ET.SubElement(root,'size')
        sizes.text = '\n\t\t'
        sizes.tail = '\n\t'
        width = ET.SubElement(sizes, 'width')
        width.text = str(img_w)
        width.tail = '\n\t\t'
        height = ET.SubElement(sizes, 'height')
        height.text = str(img_h)
        height.tail = '\n\t\t'
        depth = ET.SubElement(sizes, 'depth')
        depth.text = str(img_d)
        depth.tail = '\n\t'

        # 创建segmented子节点
        segmented = ET.SubElement(root, 'segmented')
        segmented.text = '0'
        segmented.tail = '\n\t'

        for labeldict in labeldicts:
            # 创建object子节点
            objects = ET.SubElement(root, 'object')
            objects.text = '\n\t\t'
            objects.tail = '\n'

            name = ET.SubElement(objects, 'name')
            name.text = labeldict['name']
            name.tail = '\n\t\t'

            pose = ET.SubElement(objects, 'pose')
            pose.text = 'Unspecified'
            pose.tail = '\n\t\t'

            truncated = ET.SubElement(objects, 'truncated')  # 是否被截断，暂时设置为0
            truncated.text = '0'
            truncated.tail = '\n\t\t'

            difficult = ET.SubElement(objects, 'difficult')
            difficult.text = '0'
            difficult.tail = '\n\t\t'

            bndbox = ET.SubElement(objects,'bndbox')
            bndbox.text = '\n\t\t\t'
            bndbox.tail = '\n\t'

            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(int(labeldict['xmin']))
            xmin.tail = '\n\t\t\t'

            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(int(labeldict['ymin']))
            ymin.tail = '\n\t\t\t'

            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(int(labeldict['xmax']))
            xmax.tail = '\n\t\t\t'

            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(int(labeldict['ymax']))
            ymax.tail = '\n\t\t\t'
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8')

    def voc2yolo(self, classes):
        """voc数据集转yolo格式"""

        print('开始转换数据集格式。。。')
        with open(f'{self.dataset_path}/classes.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(classes))  # 保存为yolo格式类别文件

        os.makedirs(f'{self.dataset_path}/labels', exist_ok=True)
        imgs = os.listdir(self.image_path)
        for img in tqdm(imgs):
            img_id = os.path.splitext(img)[0]
            self.xml2txt(img_id, classes)  # 生成labels文件夹下的txt

    def yolo2voc(self):
        """yolo数据集转voc格式"""

        classes_txt = f'{self.dataset_path}/classes.txt'
        with open(classes_txt, 'r') as f:
            classes = f.readlines()

        label_files = os.listdir(self.txt_label_path)

        for label_file in label_files:
            img_name = os.path.splitext(label_file)[0]
            try:
                img = np.array(Image.open(f'{self.image_path}/{img_name}.jpg'))
            except FileNotFoundError:
                img = np.array(Image.open(f'{self.image_path}/{img_name}.png'))
            img_h, img_w, img_d = img.shape[0], img.shape[1], img.shape[2]

            with open(f'{self.txt_label_path}/{label_file}', 'r') as f:
                contents = f.readlines()
            labeldicts = []
            for content in contents:
                content = content.strip('\n').split()
                x = float(content[1])*img_w
                y = float(content[2])*img_h
                w = float(content[3])*img_w
                h = float(content[4])*img_h

                # 坐标的转换，x_center y_center width height -> xmin ymin xmax ymax
                new_dict = {'name': classes[int(content[0])],
                            'difficult': '0',
                            'xmin': x-w/2,
                            'ymin': y-h/2,
                            'xmax': x+w/2,
                            'ymax': y+h/2
                            }
                labeldicts.append(new_dict)
            self.txt2xml(img_name, img_w, img_h, img_d, f'{self.xml_label_path}/{img_name}.xml', labeldicts)

    def get_data(self, classes: list, file_data, annotation_type: str='voc', class_names=None) -> list:
        if annotation_type == 'voc': # xml文件中每个object标签的格式为：<object><name>类别名</name></object>
            tree = ET.parse(file_data)
            root = tree.getroot()
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    classes.append(cls)
        elif annotation_type == 'yolo': # txt文件中每行的格式为：类别号 x1 y1 x2 y2
            assert class_names is not None, 'class_names must be not None'
            for line in file_data:
                serial_number = line.strip().split(' ')[0]
                cls = f'{serial_number}:{class_names[int(serial_number)]}'
                if cls not in classes:
                    classes.append(cls)
        return classes

    def get_labels(self, annotation_type='voc'):
        """获取数据集中的类别和类别数"""
        assert annotation_type in ['voc', 'yolo'], 'annotation_type must be voc or yolo'

        if annotation_type == 'voc':
            annotation_path = self.xml_label_path
            class_names = None
        elif annotation_type == 'yolo':
            annotation_path = self.txt_label_path
            # class_names = None

            # # 读取对应yaml文件中的类别名
            # with open(os.path.join(self.dataset_path, f'{self.dataset_name}.yaml'), 'r', encoding='utf-8') as f:
            #     class_names = yaml.load(f, Loader=yaml.FullLoader)['names']

            # 读取对应yaml文件中的类别名
            with open(os.path.join(self.dataset_path, 'classes.txt'), 'r', encoding='utf-8') as f:
                # class_names = yaml.load(f, Loader=yaml.FullLoader)['names']
                class_names = f.readlines()

        classes = []
        file_list = os.listdir(annotation_path)  # 文件夹下所有文件的文件名列表
        for file_name in file_list:
            file_path = os.path.join(annotation_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                classes = self.get_data(classes, f, annotation_type, class_names)
        return classes

    # 批量修改VOC数据集中xml的标签名称
    def change_xml_label(self, old_class, new_class):
        file_names = os.listdir(self.xml_label_path)
        count = 0
        for file in file_names:
            if file.endswith('xml'):
                file = os.path.join(self.xml_label_path, file)
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

    def data_formatting(self, train_ratio=0.7, val_ratio=0.2, seed=None, annotation_type='voc'):
        """数据格式化"""

        self.data_divide(train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        classes = self.get_labels(annotation_type=annotation_type)
        if annotation_type == 'voc':
            self.voc2yolo(classes)
        elif annotation_type == 'yolo':
            self.yolo2voc()

        self.create_yaml(classes)
        print(f'总共有{len(classes)}个类别：{classes}')


if __name__ == '__main__':
    dataset_path = 'datasets/detention'

    data_process = Data_process(dataset_path)
    data_process.data_formatting(train_ratio=0.8, val_ratio=0.2, seed=0, annotation_type='voc')
