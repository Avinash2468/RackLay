import os
import random

import PIL.Image as pil
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import numpy as np

import torch.utils.data as data

from torchvision import transforms
import torch

def pil_loader(path):
    with open(path, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('RGB')

def npy_loader(path):
    return np.load(path,allow_pickle=True)
    
def process_topview(topview_temp, size,num_racks):
    topview_final = np.zeros((num_racks,size, size))
    for i in range(num_racks):
        topview = topview_temp[i,:,:]
        topview = cv2.resize(topview, dsize=(size, size), interpolation=cv2.INTER_NEAREST)
        topview = np.array(topview)
        topview_n = np.zeros(topview.shape)
        topview_n[topview == 115] = 1  # [1.,0.] # Update the GT value
        topview_n[topview == 255] = 2  # [1.,0.]
        topview_final[i,:,:] = topview_n
    return np.asarray(topview_final)

def resize_topview(topview, size):
    topview = topview.convert("1")
    topview = topview.resize((size, size), pil.NEAREST)
    topview = topview.convert("L")
    topview = np.array(topview)
    return topview

class Loader(data.Dataset):
    def __init__(self, opt, filenames, is_train = True):
        super(Loader, self).__init__()

        self.opt = opt
        self.data_path = self.opt.data_path
        self.filenames = filenames
        self.is_train = is_train
        self.height = self.opt.height
        self.width = self.opt.width
        self.interp = pil.ANTIALIAS
        # self.loader = pil_loader
        self.loader = npy_loader
        self.to_tensor = transforms.ToTensor()
        
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = transforms.Resize(
            (self.height, self.width), interpolation=self.interp)

    def preprocess(self, inputs, color_aug):

        inputs["color"] = color_aug(self.resize(inputs["color"]))

        for key in inputs.keys():
            if key != "color" and "discr" not in key:
                inputs[key] = process_topview(inputs[key], self.opt.occ_map_size,self.opt.num_racks)
                inputs[key] = torch.from_numpy(inputs[key])
            else:
                inputs[key] = self.to_tensor(inputs[key])


    def get_image_path(self, root_dir, frame_index):
        img_path = os.path.join(root_dir, "%06d.jpg" % int(frame_index))
        #img_path = os.path.join(root_dir, "front" + "%06d.npy" % int(frame_index))

        return img_path

    def get_top_path(self, root_dir, frame_index):
        img_path = os.path.join(root_dir, "top"+"%06d.npy" % int(frame_index))

        return img_path

    def get_front_path(self, root_dir, frame_index):
        img_path = os.path.join(root_dir, "front"+"%06d.npy" % int(frame_index))

        return img_path

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        
        # do_color_aug = True
        # do_flip = True

        frame_index = self.filenames[index]  # .split()
        # check this part from original code if the dataset is changed
        folder = self.opt.data_path

        inputs["color"] = self.get_color(folder+"Images/", frame_index, do_flip)
        if(self.opt.type == "both" or self.opt.type == "topview"):
            inputs["topview"] = self.get_top(folder+"topLayouts/", frame_index, do_flip)
        if(self.opt.type == "both" or self.opt.type == "frontview"):
            inputs["frontview"] = self.get_front(folder+"frontLayouts/", frame_index, do_flip)

        
        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        return inputs

    def get_color(self, folder, frame_index, do_flip):
        color = pil_loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        #color = Image.fromarray(color.astype('uint8'), 'RGB')
        return color

    def get_top(self, folder, frame_index, do_flip):

        tv = self.loader(self.get_top_path(folder, frame_index))
     
        if do_flip:
            tv_copy = np.zeros_like(tv)
            for i in range(self.opt.num_racks):
                topview = tv[i,:,:]
                tv_copy[i,:,:] = np.fliplr(topview)
            tv = tv_copy

        return tv

    def get_front(self, folder, frame_index, do_flip):

        tv = self.loader(self.get_front_path(folder, frame_index))
     
        if do_flip:
            tv_copy = np.zeros_like(tv)
            for i in range(self.opt.num_racks):
                frontview = tv[i,:,:]
                tv_copy[i,:,:] = np.fliplr(frontview)
            tv = tv_copy
        return tv

