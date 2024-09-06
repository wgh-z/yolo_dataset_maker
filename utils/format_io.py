def xyxy2xywhn(box, img_w, img_h):  # size:(原图w,原图h) , box:(xmin,ymin,xmax,ymax)
    x = (box[0] + box[2]) / 2.0  # 中心点x坐标
    y = (box[1] + box[3]) / 2.0  # 中心点y坐标
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x / img_w  # 归一化比例
    y = y / img_h
    w = w / img_w
    h = h / img_h
    return (x, y, w, h)
