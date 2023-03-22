import torch
import torch.optim as optim
from PIL.ImageFilter import SMOOTH
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import os
import warnings
from tqdm import tqdm
import numpy as np
from time import time

from networks.DeepLabV3 import DeepLabv3_plus
from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
import torch.nn.functional as F


# from test import TTAFrame


def iou(img_true, img_pred):
    img_pred = (img_pred > 0).float()
    i = (img_true * img_pred).sum()
    u = (img_true + img_pred).sum()
    return i / u if u != 0 else u


def iou_metric(imgs_pred, imgs_true):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    for i in range(num_images):
        if imgs_true[i].sum() == imgs_pred[i].sum() == 0:
            scores[i] = 1
        else:
            # iou_thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
            # scores[i] = (iou_thresholds <= iou(imgs_true[i], imgs_pred[i])).mean()
            scores[i] = iou(imgs_true[i], imgs_pred[i])
    return scores.mean()


def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)
    size.append(N)
    return ones.view(*size)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss

class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        #   weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, SMOOTH=1e-6):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch


# Numpy version
# Well, it's the same function, so I'm going to omit the comments

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded  # Or thresholded.mean()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # SHAPE = (512, 512)
    # 输入图像大小
    SHAPE = (512, 512)
    train_root = './dataset/GIDNew/'
    imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(train_root))
    trainlist = map(lambda x: x[:-8], imagelist)
    trainlist = list(trainlist)

    val_root = './dataset/GIDNew_val/'
    imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(val_root))
    vallist = map(lambda x: x[:-8], imagelist)
    vallist = list(vallist)

    # DeepG6效果较好
    # 模型名称--------------------------------------------------------------------------------------------
    NAME = 'DeepLabV3_GID_shaixuan'
    BATCHSIZE_PER_CARD = 8  # 每个显卡给batchsize给8

    solver = MyFrame(DinkNet34, dice_bce_loss, 1e-3)#网络 损失函数 学习率
    # solver.load('./weights/test.th')

    train_batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
    # val_batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD * 2
    val_batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
    train_dataset = ImageFolder(trainlist, train_root)
    val_dataset = ImageFolder(vallist, val_root)

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batchsize,
        shuffle=True,
        # shuffle=True,
        num_workers=0,
        drop_last=False)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batchsize,
        shuffle=True,
        num_workers=0,
        drop_last=False)

    mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()
    device = torch.device('cuda:0')
    no_optim = 0
    total_epoch = 150
    train_epoch_best_loss = 300.

    test_loss = 0
    # criteon = nn.CrossEntropyLoss().to(device)
    criteon = DiceLoss()
    iou_criteon = SoftIoULoss(2)
    # scheduler = solver.lr_strategy()  这个学习率调整策略是我加的，还没用，只用了原始的，感兴趣的可以试试
    for epoch in range(1, total_epoch + 1):
        print('---------- Epoch:' + str(epoch) + ' ----------')
        # scheduler.step() 对应上面的学习率策略，须同时打开
        # print('lr={:.6f}'.format(scheduler.get_lr()[0])) 输出上面的学习率策略，须同时打开
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        print('Train:')
        for img, mask in tqdm(data_loader_iter, ncols=20, total=len(data_loader_iter)):
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(data_loader_iter)

        val_data_loader_num = iter(val_data_loader)
        test_epoch_loss = 0
        test_mean_iou = 0
        val_pre_list = []
        val_mask_list = []
        print('Validation:')
        for val_img, val_mask in tqdm(val_data_loader_num, ncols=20, total=len(val_data_loader_num)):
            val_img, val_mask = val_img.to(device), val_mask.cpu()
            # val_img, val_mask = val_img.to(device), val_mask.cuda()
            val_mask[np.where(val_mask > 0)] = 1
            val_mask = val_mask.squeeze(0)
            predict = solver.test_one_img(val_img)
            predict_temp = torch.from_numpy(predict).unsqueeze(0)
            predict_use = V(predict_temp.type(torch.FloatTensor), volatile=True)
            val_use = V(val_mask.type(torch.FloatTensor), volatile=True)
            test_epoch_loss += criteon.forward(predict_use, val_use)
            predict_use = predict_use.squeeze(0)
            predict_use = predict_use.unsqueeze(1)
            predict_use[predict_use >= 0.5] = 1
            predict_use[predict_use < 0.5] = 0
            predict_use = predict_use.type(torch.LongTensor)
            val_use = val_use.squeeze(1).type(torch.LongTensor)
            test_mean_iou += iou_pytorch(predict_use, val_use)

        batch_iou = test_mean_iou / len(val_data_loader_num)
        val_loss = test_epoch_loss / len(val_data_loader_num)

        mylog.write('********************' + '\n')
        mylog.write('--epoch:' + str(epoch) + '  --time:' + str(int(time() - tic)) + '  --train_loss:' + str(
            train_epoch_loss) + ' --val_loss:' + str(val_loss) + ' --val_iou:' + str(
            batch_iou) + '\n')
        print('--epoch:', epoch, '  --time:', int(time() - tic), '  --train_loss:', train_epoch_loss,
              ' --val_loss:', val_loss, ' --val_iou:', batch_iou)
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
            print('no_optim 的值为：' + str(no_optim))
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('weights/' + NAME + '.th')
        # if no_optim > 6:
        #     print(mylog, 'early stop at %d epoch' % epoch)
        #     print('early stop at %d epoch' % epoch)
        #     break
        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break
            solver.load('weights/' + NAME + '.th')
            solver.update_lr(5.0, factor=True, mylog=mylog)
        mylog.flush()


    print(mylog, 'Finish!')
    print('Finish!')
    mylog.close()
