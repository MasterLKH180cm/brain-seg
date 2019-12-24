# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:37:26 2019

@author: user
"""
from PIL import Image
import nibabel as nib
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
#import torchvision.transforms.functional as TF
import torch.optim as optim
import torch.nn.functional as F
import glob
import pdb
import sys
import os
import traceback
import argparser_brain_seg
import matplotlib.pyplot as plt
import cv2
from skimage.filters import gaussian
from skimage.util import random_noise
from skimage.transform import rotate, resize
from numpy import fliplr, flipud
import models
from misc import runningScore
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('===> build dataset ...')
    train_set = MyDataset(r'E:\brain_seg\train_subset\train_img', r'E:\brain_seg\train_subset\train_label',device)
    test_set = MyDataset(r'E:\brain_seg\train_subset\test_img', r'E:\brain_seg\train_subset\test_label',device)
#    print(train_set[0])
    train_loss = []
    print('===> build dataloader ...')
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=args.train_batch,num_workers=args.workers,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=args.train_batch,num_workers=args.workers,shuffle=True)
    print('===> basic setting ...')
    
    model = models.seg_model(args).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_iou = 0
    
    unet = models.UNet().load_state_dict(torch.load('unet.pt'))
    num_cls = 2
    running_metrics = runningScore(num_cls)
    iou_score = []
    
    for epoch in range(1, args.epoch+1):
        print('===> set training mode ...')
        
        model.train()
        
        for idx, (imgs, labels) in enumerate(train_loader):
            
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(0, idx+1, len(train_loader))
            
#            seg_loss = Variable(torch.tensor([0.]),requires_grad=True)
            ''' move data to gpu '''
#            imgs = Variable(imgs,requires_grad=True).to(device)
#            labels = Variable(labels,requires_grad=False).to(device)
            imgs, labels = imgs.to(device), labels.to(device)
#            print('===> batch{0}'.format(idx))
            pred = model(imgs)
            
            seg_loss = loss(pred,labels)
            
#            optimizer.zero_grad()
            
#            print('===> backward')
            
            seg_loss.backward() 
#            print('===> step')              
            optimizer.step()
            for i in range(args.train_batch):
                cv2.imwrite(r'E:\brain_seg\\test{0}.png'.format(i),pred[i].argmax(0).cpu().numpy().astype(np.uint8)*255)
#            print('===> append loss')
            train_loss += [seg_loss.data]
            train_info += 'Segmentation loss: {:.12f}'.format(seg_loss.data.cpu().numpy())
            print(train_info)#, end="\r"
        model.eval()
        for idx, (imgs, labels) in enumerate(test_loader):
                ''' evaluate the model '''
                pred = model(imgs.to(device)).cpu()
                iou = mean_iou_score(pred, labels)
                iou_score += [iou.item()]
    #            writer.add_scalar('val_acc', acc, iters)
                print('Epoch: [{}] iou:{}'.format(epoch, iou))
                
                ''' save best model '''
                if iou > best_iou:
                    save_model(model, os.path.join(args.save_dir, 'model_best.h5'))
                    best_iou = iou

        save_model(model, os.path.join(args.save_dir, 'model'+str(epoch)+'.h5'))

    plt.figure()
    plt.plot(range(1,len(train_loss)+1),train_loss,'-')
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("training loss")
    plt.show()
    plt.savefig(os.path.join(args.save_dir,'Loss.jpg'))
    
    plt.figure()
    plt.plot(range(1,len(iou_score)+1),iou_score,'-')
    plt.xlabel("epoch")
    plt.ylabel("iou_score")
    plt.title("iou_score")
    plt.show()
    plt.savefig(os.path.join(args.save_dir,'iou_score.jpg'))


def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)  





class MyDataset(Dataset):
    def __init__(self, image_paths, label_paths,device, train=True):

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.files = os.listdir(self.image_paths)
        self.lables = os.listdir(self.label_paths)
        
    def transform(self, image, mask):
        # Resize
        
        image = cv2.resize(image,(512,512))
        mask = cv2.resize(mask,(512,512))
        
        

        # Random horizontal flipping
#        if random.random() > 0.5:
#            image = flipud(image)
#            mask = flipud(mask)

#        # Random vertical flipping
#        if random.random() > 0.5:
#            image = fliplr(image)
#            mask = fliplr(mask)
        image = np.stack([image, gaussian(image), random_noise(image)], axis=0)
#        mask = np.stack([mask, mask, mask], axis=0)
        
        # Transform to tensor
        image = torch.from_numpy(image.copy()).to(dtype = torch.float)
        mask = torch.from_numpy(mask.copy()).to(dtype = torch.long)
        
        return image, mask
    def __len__(self):
       
        return len(self.files)
    def __getitem__(self,idx):
        
        img_name = self.files[idx]
        label_name = self.lables[idx]
#        image = nib.load(os.path.join(self.image_paths,img_name)).get_fdata()
#        image = image / np.max(image) * 255
#        mask = nib.load(os.path.join(self.label_paths,label_name)).get_fdata()
#        mask = mask / np.max(mask) * 255
#        image = image#.astype('uint8')
#        mask = mask#.astype('uint8')
#        x, y = self.transform(image,mask)
        img, mask = cv2.imread(os.path.join(self.image_paths,img_name), cv2.IMREAD_GRAYSCALE), cv2.imread(os.path.join(self.label_paths,label_name), cv2.IMREAD_GRAYSCALE)
        return self.transform(img,mask)

def mean_iou_score(pred, labels, num_classes=2):
    '''
    Compute mean IoU score over 9 classes
    '''
    mean_iou = 0
    for i in range(num_classes):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / num_classes
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc


if __name__ == '__main__':
    args = argparser_brain_seg.arg_parse()
    
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
