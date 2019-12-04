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
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.optim as optim
import glob
import pdb
import sys
import os
import traceback
import argparser_brain_seg
import matplotlib.pyplot as plt
import sklearn
import skimage
import models

def main(args):
# =============================================================================
#     imgs_list = glob.glob(r'E:\brain_seg\train_subset\image\*.nii.gz')
#     labels_list = glob.glob(r'E:\brain_seg\train_subset\label\*.nii.gz')
#     imgs = []
#     labels = []
#     for i, j in zip(imgs_list,labels_list[0:5]):
#         imgs += [nib.load(i).get_fdata()]
#         
#         imgs[-1] = imgs[-1]/np.max(imgs[-1])
# #        fig, ax = plt.subplots()
# #        im = ax.imshow(imgs[-1][:,:,0],cmap = 'gray')
#         labels += [nib.load(j).get_fdata()]
# =============================================================================
    train_set = MyDataset(r'E:\brain_seg\train_subset\image', r'E:\brain_seg\train_subset\label')
    test_set = MyDataset(r'E:\brain_seg\test\image', r'E:\brain_seg\test\label')
#    print(train_set[0])
    train_loss = []
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.seg_model(args).to(device)
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_acc = 0
    for epoch in range(1, args.epoch+1):
        print('===> set training mode ...')
        
        model.train()
        
        for idx, (imgs, labels) in enumerate(train_loader):
            
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))


            ''' move data to gpu '''
#            imgs = (imgs - 127.5) / 127.5
#            imgs = Variable(imgs,requires_grad=True).to(device)
#            labels = Variable(labels,requires_grad=True).to(device)
            
            ''' compute loss, backpropagation, update parameters '''
            seg_loss = loss(model(imgs))
            optimizer.zero_grad()         
            seg_loss.backward()               
            optimizer.step()

            
#            imageio.imwrite(args.save_dir + '\{:0>5d}.png'.format(count), fake_imgs[0].transpose(0,2).transpose(0,1).detach().cpu().numpy().astype(np.uint8))          
#            count += 1
            train_loss += [seg_loss.item()]
            train_info += 'Segmentation loss: {:.12f}'.format(seg_loss.data.cpu().numpy())
            print(train_info)#, end="\r"
        if epoch%args.val_epoch == 0:
                ''' evaluate the model '''
                acc, iou = evaluate(model, val_loader) 
                iou_score += [acc]
    #            writer.add_scalar('val_acc', acc, iters)
                print('Epoch: [{}] ACC:{}'.format(epoch, acc))
                
                ''' save best model '''
                if acc > best_acc:
                    save_model(model, os.path.join(args.save_dir, 'model_best.h5'))

                    best_acc = acc
                    best_iou = iou
#        imsave(args.save_dir + '\{:0>5d}.png'.format(epoch), (generator(noise)[0].transpose(0,2).transpose(0,1).detach().cpu().numpy() *127.5 + 127.5).astype(np.uint8))
        total_iter += (idx+1)        
        save_model(model, os.path.join(args.save_dir, 'model'+str(epoch)+'.h5'))




def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)  

class MyDataset(Dataset):
    def __init__(self, image_paths, label_paths, train=True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.files = os.listdir(self.image_paths)
        self.lables = os.listdir(self.label_paths)
        
    def transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(512, 512))
        image = [resize(img) for img in image]
        mask = [resize(img) for img in mask]

        

        # Random horizontal flipping
        if random.random() > 0.5:
            image = [TF.hflip(img) for img in image]
            mask = [TF.hflip(img) for img in mask]

        # Random vertical flipping
        if random.random() > 0.5:
            image = [TF.vflip(img) for img in image]
            mask = [TF.vflip(img) for img in mask]

        # Transform to tensor
        image = [TF.to_tensor(img) for img in image]
        mask = [TF.to_tensor(img) for img in mask]
        return image, mask
    def __len__(self):
       
        return len(self.image_paths)
    def __getitem__(self,idx):
        img_name = self.files[idx]
        label_name = self.lables[idx]
        image = nib.load(os.path.join(self.image_paths,img_name)).get_fdata()
        image = image / np.max(image) * 255
        mask = nib.load(os.path.join(self.label_paths,label_name)).get_fdata()
        mask = mask / np.max(mask) * 255
        image = [Image.fromarray(img.astype('uint8'),mode='L') for img in image]
        mask = [Image.fromarray(label.astype('uint8'),mode='L')  for label in mask]
        x, y = self.transform(image, mask)
        return x, y





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
