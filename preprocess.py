# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:37:26 2019

@author: user
"""
from PIL import Image
import nibabel as nib
import numpy as np
import cv2
import random
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
#import torchvision.transforms.functional as TF
import torch.optim as optim
import glob
import pdb
import sys
import os
import traceback
import argparser_brain_seg
import matplotlib.pyplot as plt
import sklearn
from skimage.io import imsave
from skimage.filters import gaussian
from skimage.util import random_noise
from skimage.transform import rotate, resize
from numpy import fliplr, flipud
import models
from misc import runningScore
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# =============================================================================
#     imgs_list = glob.glob(r'E:\brain_seg\train_subset\image\*.nii.gz')
#     labels_list = glob.glob(r'E:\brain_seg\train_subset\label\*.nii.gz')
#     imgs = []
#     labels = []
#     for i, j in zip(imgs_list,labels_list[0:5]):
#         imgs += [nib.load(i).get_fdata()]
#         print(imgs[-1].shape)
#         imgs[-1] = imgs[-1]/np.max(imgs[-1])
# #        fig, ax = plt.subplots()
# #        im = ax.imshow(imgs[-1][:,:,0],cmap = 'gray')
#         labels += [nib.load(j).get_fdata()]
# =============================================================================
    print('===> build dataset ...')
    train_set = MyDataset(r'E:\brain_seg\train_subset\image', r'E:\brain_seg\train_subset\label',device)
    test_set = MyDataset(r'E:\brain_seg\test\image', r'E:\brain_seg\test\label',device)

    print('===> build dataloader ...')
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=args.train_batch,num_workers=args.workers,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=args.train_batch,num_workers=args.workers,shuffle=True)
    print('===> basic setting ...')
    
    count = 0
#    imgs = []
#    label = []
    for idx, (imgs_filename, labels_filename) in enumerate(train_loader):
        
        for img_name, label_name in zip(imgs_filename, labels_filename):
            
            image = nib.load(img_name).get_fdata()
            image = image / np.max(image) * 255
            mask = nib.load(label_name).get_fdata()
#            mask = mask / np.max(mask) * 255
            image = image.astype('uint8')
            mask = mask.astype('uint8')
#            imgs += [image]
#            label += [mask]
            for i,j in zip(np.transpose(image,(2,0,1)),np.transpose(mask,(2,0,1))):
                count+=1
                print(count,end='\r')
                cv2.imwrite(r'E:\brain_seg\train_subset\train_img\{:0>6d}.png'.format(count),i)
                cv2.imwrite(r'E:\brain_seg\train_subset\train_label\{:0>6d}.png'.format(count),j)
    for idx, (imgs_filename, labels_filename) in enumerate(test_loader):
        
        for img_name, label_name in zip(imgs_filename, labels_filename):
            
            image = nib.load(img_name).get_fdata()
            image = image / np.max(image) * 255
            mask = nib.load(label_name).get_fdata()
#            mask = mask / np.max(mask) * 255
            image = image.astype('uint8')
            mask = mask.astype('uint8')
#            imgs += [image]
#            label += [mask]
            for i,j in zip(np.transpose(image,(2,0,1)),np.transpose(mask,(2,0,1))):
                count+=1
                print(count,end='\r')
                cv2.imwrite(r'E:\brain_seg\train_subset\test_img\{:0>6d}.png'.format(count),i)
                cv2.imwrite(r'E:\brain_seg\train_subset\test_label\{:0>6d}.png'.format(count),j)






class MyDataset(Dataset):
    def __init__(self, image_paths, label_paths,device, train=True):

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.files = os.listdir(self.image_paths)
        self.lables = os.listdir(self.label_paths)
        
    def __len__(self):
       
        return len(self.files)
    def __getitem__(self,idx):
        
        img_name = self.files[idx]
        label_name = self.lables[idx]
        return os.path.join(self.image_paths,img_name), os.path.join(self.label_paths,label_name)





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
