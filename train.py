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
from ranger import Ranger
import torchvision.models as tvmodels
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    
#    train_img_list = glob.glob(r'.\train_subset\image\*.nii.gz')
#    train_seg_list = glob.glob(r'.\train_subset\label\*.nii.gz')
#    val_img_list = glob.glob(r'.\test\image\*.nii.gz')
#    val_seg_list = glob.glob(r'.\test\label\*.nii.gz')
    print('===> build dataset ...')
    train_set = MyDataset(r'E:\brain_seg\train_subset\image', r'E:\brain_seg\train_subset\label',device)
    test_set = MyDataset(r'E:\brain_seg\test\image', r'E:\brain_seg\test\label',device)
#    print(train_set[0])
    train_loss = []
    print('===> build dataloader ...')
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=args.train_batch,num_workers=args.workers,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=args.train_batch,num_workers=args.workers,shuffle=False)
    print('===> basic setting ...')
    
#    model = tvmodels.segmentation.deeplabv3_resnet101(num_classes = 2).to(device)
    model = models.UNet(in_channels=3, init_features=32, out_channels=1).to(device)#
    model.load_state_dict(torch.load('unet.pt'))
#    print(model)
    loss_1 = DiceLoss()
    loss_2 = BCELoss()
#    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = Ranger(model.parameters(), lr=args.lr)
    best_iou = 0
    
    
    num_cls = 2
    running_metrics = runningScore(num_cls)
    iou_score = []
    threshold = torch.tensor([0.5]).to(device,dtype = torch.float)
    for epoch in range(1, args.epoch+1):
        print('===> set training mode ...')
        
        model.train()
        epoch_loss = 0
        for idx, (imgs, labels) in enumerate(train_loader):
            
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
            imgs = imgs.transpose(1,3).transpose(2,3)
            labels = labels.transpose(1,3).transpose(2,3)
            img_height = imgs.size()[1]
            
            imgs = torch.cat((imgs[:,0,:,:].unsqueeze(0),imgs,imgs[:,-1,:,:].unsqueeze(0)),dim=1)
            seg_loss = torch.tensor(0.,requires_grad=True).to(device)
            for img_idx in range(img_height):
                
                label = labels[:,img_idx,:,:].to(device,dtype = torch.long)

                img = imgs[:,img_idx:img_idx+3,:,:].to(device,dtype = torch.float)
                
                pred = model(img)
                
                seg_loss = loss_1(pred,label)/img_height + loss_2(pred,label)/img_height
                seg_loss.backward()
                epoch_loss = seg_loss.item()
                torch.cuda.empty_cache()
                print('loss = '+ str(seg_loss.item()), end="\r")
            
            optimizer.step()
            optimizer.zero_grad()
            
            
            train_loss += [epoch_loss]
            train_info += 'Segmentation loss: {:.12f}'.format(seg_loss.data.cpu().numpy())
            print(train_info)
#            print("                                                                         ", end="\r")
        
        
        if epoch%args.val_epoch == 0:
            model.eval()
            
            iou = 0.
            for idx, (imgs, labels) in enumerate(test_loader):
                pred = []
                
                imgs = imgs.transpose(1,3).transpose(2,3)
            
    #            1 * 66 * 128 * 128
                labels = labels.transpose(1,3).transpose(2,3)
                tmp = imgs.size()[1]
                imgs = torch.cat((imgs[:,0,:,:].unsqueeze(0),imgs,imgs[:,-1,:,:].unsqueeze(0)),dim=1)
                
                for img_idx in range(tmp):
                    output = model(imgs[:,img_idx:img_idx+3,:,:].to(device,dtype = torch.float))
                    output = (output>threshold).to(device,dtype = torch.float)*1
                    output = output.cpu().detach().numpy()
#                    print(np.unique(output))
                    pred += [output]
                    
                iou += mean_iou_score(np.concatenate(pred,axis=1),labels.detach().numpy())
            iou /= (idx+1)
            iou_score += [iou]    
            print('Epoch: [{}] iou:{}'.format(epoch, iou))
            if iou > best_iou:
                save_model(model, os.path.join(args.save_dir, 'model_best.h5'))
                best_iou = iou
                
        save_model(model, os.path.join(args.save_dir, 'model'+str(epoch)+'.h5'))
        
        plt.plot(range(1,len(train_loss)+1),train_loss,'-')
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title("training loss")
        plt.savefig(os.path.join(args.save_dir,'Loss.png'))
        plt.clf()
        
        plt.plot(range(1,len(iou_score)+1),iou_score,'-')
        plt.xlabel("epoch")
        plt.ylabel("iou_score")
        plt.title("iou_score")
        plt.savefig(os.path.join(args.save_dir,'iou_score.png'))
        plt.clf()

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)  





class MyDataset(Dataset):
    def __init__(self, image_paths, label_paths,device, train=True):

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.files = os.listdir(self.image_paths)#[:2]
        self.lables = os.listdir(self.label_paths)#[:2]
        self.device = device
        
    def transform(self, image, mask):
        # Resize
        
        image = cv2.resize(image,(128,128))
        mask = cv2.resize(mask,(128,128))
        
    
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
        image = nib.load(os.path.join(self.image_paths,img_name)).get_fdata()
        image = cv2.resize(image,(256,256))
        image = image / np.max(image) * 255.
        mask = nib.load(os.path.join(self.label_paths,label_name)).get_fdata()
        mask = cv2.resize(mask,(256,256))
        mask = mask / np.max(mask)
#        print(np.unique(mask))
        
        _,mask = cv2.threshold(mask,0,1,cv2.THRESH_BINARY)
        image, mask = torch.tensor(image), torch.tensor(mask)
        return image, mask

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
#        print(pred.shape,labels.shape)
#        print(tp_fp , tp_fn , tp)
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou
class BCELoss(nn.Module):

    def __init__(self):
        super(BCELoss, self).__init__()
       

    def forward(self, logits, true, pos_weight=None):
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(0).float(),
            true.float(),
            pos_weight=pos_weight,
        )
        return loss

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
#        assert y_pred.size() == y_true.size()
#        y_pred = y_pred.argmax(1)
#        y_pred = y_pred[:, 0].contiguous().view(-1)
#        y_true = y_true[:, 0].contiguous().view(-1)
#        print(np.unique(y_pred.cpu().detach().numpy()))
        intersection = (y_pred * y_true).sum()
        dsc = (10. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth + 8. * intersection
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
