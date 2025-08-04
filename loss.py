import torch
import torch.nn as nn

import cv2
import numpy as np
import torch.nn.functional as F

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.01, 0.7],device=torch.device('cuda:0')))   # 交叉熵损失函数 内置softmax


    def soft_dice_coeff(self, y_pred, y_true):
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        smooth = 1e-20
        i = torch.sum(y_pred)
        j = torch.sum(y_true)
        intersection = torch.sum(y_true * y_pred)
        score = (intersection + smooth) / (i + j - intersection + smooth)
        return 1 - score.mean()

    def __call__(self, y_pred, y_true, use_half_training):
        assert y_pred.size() == y_true.size(), ('y_pred.size() != y_true.size()', y_pred.size(), y_true.size())
        a = self.bce_loss(y_pred, y_true)

        assert torch.isnan(a).sum() == 0 and torch.isinf(a).sum() == 0, ('bce_loss is nan or ifinit', a)
        b = self.soft_dice_coeff(y_pred[:,1:], y_true[:,1:])

        return 0.7 * a + 0.3 * b

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score, recall_score
class SegmentationMetric(object):
    def __init__(self, numClass, ignore_labels=None):
        self.numClass = numClass
        self.ignore_labels = ignore_labels
        self.confusionMatrix = torch.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        """
        整体准确率、精度 Accuracy
        """
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelRecall(self):
        """
        查全率 Recall  每个类别的召回、查全率
        实际真正的正样本中 ( TP + FN )，预测为正例的样本数 ( TP ) 所占的⽐例
        """
        # return each category pixel accuracy (precision)
        # precision = (TP) / TP + FN
        classRecall = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)  # [H,W]->[0,1]
        return classRecall

    def classPixelF1(self):
        """
        F1 = 2*P*R/(R+P)
        """
        classF1 = 2*self.classPixelPrecision()*self.classPixelRecall()/(self.classPixelPrecision()+self.classPixelRecall())
        return classF1

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        if self.ignore_labels != None:
            union[self.ignore_labels] = 0  # 排除忽略的类别
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU<float('inf')].mean()# 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        :param imgPredict: tensor [N H W]  model-out需要argmax
        :param imgLabel:   tensor [N H W]  one-hot需要argmax
        :return: 混淆矩阵
        """
        assert imgPredict.shape == imgLabel.shape
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)  # same_shape bool_type
        if self.ignore_labels != None:
            for IgLabel in self.ignore_labels:
                mask &= (imgLabel != IgLabel)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label.type(torch.IntTensor), minlength=self.numClass ** 2)
        confusionMatrix = count.view(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU 频权交并比:为MIoU的一种提升 这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = torch.sum(self.confusionMatrix, axis=1) / torch.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (torch.sum(self.confusionMatrix, axis=1) +
                                              torch.sum(self.confusionMatrix, axis=0) - torch.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        print(imgPredict.shape,imgLabel.shape)
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass,) * 2)  # 混淆矩阵（空） torch.zeros((self.numClass, self.numClass))


    def evalus(self, imgPredict, imgLabel):
        hist = self.addBatch(imgPredict, imgLabel)
        Accuracy = self.pixelAccuracy()
        Recall = self.classPixelRecall()
        F1 = self.classPixelF1()
        mIoU = self.meanIntersectionOverUnion()
        FWIoU = self.Frequency_Weighted_Intersection_over_Union()
        evalus_res = 'Recall : {}, F1 : {}, Accuracy : {}, mIoU : {}, confusionMatrix : {}, FWIoU : {}'.format(
                                Recall.numpy(), F1.numpy(), Accuracy.numpy(), mIoU.numpy(), hist.numpy(), FWIoU.numpy())
        self.reset()
        return evalus_res

