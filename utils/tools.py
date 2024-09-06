from PIL import Image

def get_image_info(image_path):
    # 打开图像文件
    with Image.open(image_path) as img:
        # 获取图像的宽度和高度
        width, height = img.size
        # 获取图像模式，这可以告诉我们图像的通道数
        mode = img.mode
        # 根据图像模式确定通道数
        if mode in ('1', 'L', 'I;16', 'F'):
            channels = 1
        elif mode in ('P', 'I'):
            channels = 3  # 这里简化处理，实际上需要考虑调色板和其他因素
        elif mode in ('RGB', 'YCbCr'):
            channels = 3
        elif mode in ('RGBA', 'CMYK', 'RGBX'):
            channels = 4
        else:
            channels = -1  # 不常见的模式

    return width, height, channels
