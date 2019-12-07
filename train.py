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
#    print(train_set[0])
    train_loss = []
    print('===> build dataloader ...')
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=args.train_batch,num_workers=args.workers,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=args.train_batch,num_workers=args.workers,shuffle=True)
    print('===> basic setting ...')
    
    model = models.seg_model(args).to(device)
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_dice = 0
    
    
    num_cls = 2
    running_metrics = runningScore(num_cls)
    
    for epoch in range(1, args.epoch+1):
        print('===> set training mode ...')
        
        model.train()
        
        for idx, (imgs_filename, labels_filename) in enumerate(train_loader):
            imgs = []
            labels = []
            for img_name, label_name in zip(imgs_filename, labels_filename):
                image = nib.load(img_name).get_fdata()
                image = image / np.max(image) * 255
                mask = nib.load(label_name).get_fdata()
                mask = mask / np.max(mask) * 255
                image = image.astype('uint8')
                mask = mask.astype('uint8')
                image, mask = transform(image,mask)
                imgs += [image]
                labels+= [mask]
            imgs = torch.Tensor(np.array(imgs))
            labels = torch.Tensor(np.array(labels))
            print(imgs.shape)
            imgs = imgs.transpose(1,3).transpose(2,3)
            labels = labels.transpose(1,3).transpose(2,3)
            train_info = str(device)+'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
            imgs, labels = imgs.squeeze().unsqueeze(1), labels.squeeze().unsqueeze(1)
            imgs = torch.cat([imgs,imgs,imgs], dim=1)
#            print(imgs.shape)
            loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(imgs, labels),batch_size=50,num_workers=args.workers,shuffle=False)
            ''' move data to gpu '''
#            imgs = (imgs - 127.5) / 127.5
#            imgs = Variable(imgs,requires_grad=True).to(device)
#            labels = Variable(labels,requires_grad=True).to(device)
            seg_loss = 0
            ''' compute loss, backpropagation, update parameters '''
            for img, label in loader:
                seg_loss = loss(model(img.to(device, dtype=torch.float)),label.to(device, dtype=torch.float))
#                print(seg_loss)
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
                acc, iou = evaluate(model, test_loader) 
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


def transform(image, mask):
    # Resize
    
    image = resize(image,(512,512))
    mask = resize(mask,(512,512))

    

    # Random horizontal flipping
    if random.random() > 0.5:
        image = flipud(image)
        mask = flipud(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = fliplr(image)
        mask = fliplr(mask)

    # Transform to tensor
#    image = torch.from_numpy(image.copy())
#    mask = torch.from_numpy(mask.copy())
    return image, mask


class MyDataset(Dataset):
    def __init__(self, image_paths, label_paths,device, train=True):

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.files = os.listdir(self.image_paths)
        self.lables = os.listdir(self.label_paths)
        
    def transform(self, image, mask):
        # Resize
        
        image = resize(image,(512,512))
        mask = resize(mask,(512,512))

        

        # Random horizontal flipping
        if random.random() > 0.5:
            image = flipud(image)
            mask = flipud(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = fliplr(image)
            mask = fliplr(mask)

        # Transform to tensor
        image = torch.from_numpy(image.copy())
        mask = torch.from_numpy(mask.copy())
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
