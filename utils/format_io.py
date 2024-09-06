"""格式转换"""

import xml.etree.ElementTree as ET


def xyxy2xywh(xyxy):
    """# xyxy: (xmin,ymin,xmax,ymax)"""
    xywh = (
        (xyxy[0] + xyxy[2]) / 2.0,  # 中心点x坐标
        (xyxy[1] + xyxy[3]) / 2.0,  # 中心点y坐标
        xyxy[2] - xyxy[0],
        xyxy[3] - xyxy[1],
    )
    return xywh

def xywh2xyxy(xywh):
    xyxy = (
        xywh[0] - xywh[2] / 2.0,
        xywh[1] - xywh[3] / 2.0,
        xywh[0] + xywh[2] / 2.0,
        xywh[1] + xywh[3] / 2.0,
    )
    return xyxy

def xyxy2xywhn(xyxy, img_w, img_h):
    """# size:(原图w,原图h) , box:(xmin,ymin,xmax,ymax)"""

    xywh = xyxy2xywh(xyxy)
    xywhn = (
        xywh[0] / img_w,  # 归一化
        xywh[1] / img_h,
        xywh[2] / img_w,
        xywh[3] / img_h,
    )
    return xywhn

def xywhn2xyxy(xywhn, img_w, img_h):
    xywh = (
        xywhn[0] * img_w,  # 反归一化
        xywhn[1] * img_h,
        xywhn[2] * img_w,
        xywhn[3] * img_h,
        )
    xyxy = xywh2xyxy(xywh)
    return xyxy

def xml2txt(img_id, xml_label_path, txt_label_path, classes:list):
    try:
        tree = ET.parse(f'{xml_label_path}/{img_id}.xml')  # 解析xml文件
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
        # with open(f'{txt_label_path}/{img_id}.txt', 'w', encoding='utf-8') as txt_file:
        return
    except AttributeError as e:  # 忽略错误标注
        return
    else:
        with open(f'{txt_label_path}/{img_id}.txt', 'w', encoding='utf-8') as txt_file:
            txt_file.write(labels)  # class_id x y w h

def txt2xml(
        img_id,  # 图片名不带后缀
        img_w,
        img_h,
        img_c,
        xml_label_path,
        txt_label_path,
        classes
        ):
    file_path = f'{xml_label_path}/{img_id}.xml'
    with open(f'{txt_label_path}/{img_id}.txt', 'r') as f:
        labels = f.readlines()
    labeldicts = []
    for label in labels:
        label = label.strip('\n').split()

        xywhn = [float(coord) for coord in label[1:]]
        x1, x2, y1, y2 = xywhn2xyxy(xywhn, img_w, img_h)
        new_dict = {
            'name': classes[int(label[0])],
            'difficult': '0',
            'xmin': x1,
            'ymin': x2,
            'xmax': y1,
            'ymax': y2,
            }
        labeldicts.append(new_dict)

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
    filename.text = str(img_id)
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
    depth.text = str(img_c)
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
