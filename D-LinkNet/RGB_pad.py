import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
def resize_img(DATADIR, data_k):
    path = os.path.join(DATADIR, data_k)
    # 返回path路径下所有文件的名字，以及文件夹的名字，
    img_list = os.listdir(path)

    for i in img_list:
        if i.endswith('.jpg'):
            # 调用cv2.imread读入图片，读入格式为IMREAD_COLOR
            img = Image.open(path + '\\' + i )
            image = np.array(img)
            # 打印原来的图片
            plt.imshow(image, cmap=plt.gray())
            plt.show()
            channel_one = image[:, :, 0]
            channel_two = image[:, :, 1]
            channel_three = image[:, :, 2]

            # 计算img1的长度
            b = 0.5 * (256 - channel_one.shape[1])
            a = int(b)

            channel_one = np.pad(channel_one, ((a, a), (a, a)), 'constant', constant_values=(0, 0))
            channel_two = np.pad(channel_two, ((a, a), (a, a)), 'constant', constant_values=(0, 0))
            channel_three = np.pad(channel_three, ((a, a), (a, a)), 'constant', constant_values=(0, 0))
            image = np.dstack((channel_one, channel_two, channel_three))
            image = Image.fromarray(image)
            img_name = str(i) + 'sat'
            '''生成图片存储的目标路径'''
            save_path = path + 'New/'
            if os.path.exists(save_path):
                print(i)
                '''调用cv.2的imwrite函数保存图片'''
                save_img = save_path + img_name
                cv2.imwrite(save_img, image)
            else:
                os.mkdir(save_path)
                save_img = save_path + img_name
                cv2.imwrite(save_img, image)


if __name__ == '__main__':
    # 设置图片路径
    DATADIR = "D:\chzu data\crop"
    data_k = 'JPG_10'
    resize_img(DATADIR, data_k)