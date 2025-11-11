import torch
import torch.nn as nn
import numpy as np
import os
from os import listdir
from os.path import join
import random
import torch.nn.functional as F
from scipy.io import loadmat
import sys
from easydict import EasyDict
import argparse
import yaml
sys.path.append("/home/ubuntu/lcl/cvpr2025/classification")
from utils.ev_utils import *
from utils.event_utils import *

def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--validation_dataset", default='/home/lxh/LCL/cvpr2025/classification/data/dataset/cifar10_dvs')
    parser.add_argument("--training_dataset", default='/home/lxh/LCL/cvpr2025/classification/data/dataset/cifar10_dvs')
    parser.add_argument('--config_path', default='/home/ubuntu/lcl/cvpr2025/classification/configs/config_DVSCifar10.yaml')


    
    flags = parser.parse_args()
    config = load_yaml(flags.config_path)
    config.update(vars(flags))

    
    
    return EasyDict(config)

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config

def normalize_event_volume(event_volume):
    # for i in range(event_volume.shape[0]):
    #     min_val = torch.min(event_volume[i])
    #     max_val = torch.max(event_volume[i])
    #     event_volume[i] = (event_volume[i] - min_val) / (max_val - min_val)

    event_volume_flat = event_volume.view(-1)  # 展成一维
    nonzero = torch.nonzero(event_volume_flat)  # 找出非零索引
    nonzero_values = event_volume_flat[nonzero]  # 取出非零
    if nonzero_values.shape[0]:
        lower = torch.kthvalue(nonzero_values,
                                max(int(0.02 * nonzero_values.shape[0]), 1),
                                dim=0)[0][0]
        upper = torch.kthvalue(nonzero_values,
                                max(int(0.98 * nonzero_values.shape[0]), 1),
                                dim=0)[0][0]
        max_val = max(abs(lower), upper)
        event_volume = torch.clamp(event_volume, -max_val, max_val)
        event_volume /= max_val
    return event_volume

class DVSCifar10:
    def __init__(self, config, mode):
        """
        Creates an iterator over the N_Caltech101 dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or 'all' for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param augmentation: flip, shift and random window start for training
        :param mode: 'training', 'testing' or 'validation'
        """
        if mode == 'training':
            self.index = np.arange(100,1000)
            self.root = config['dataset_params']['train_data_loader']['data_path']
            self.augmentation = True
        elif mode == 'testing':
            self.index = np.arange(0,100)
            self.root = config['dataset_params']['val_data_loader']['data_path']
            self.augmentation = False
        if mode == 'validation':
            self.index = np.arange(0,100)
            self.root = config['dataset_params']['test_data_loader']['data_path']
            self.augmentation = False
        self.pixel_size = [128, 128]
        self.width = config['dataset_params']['width']
        self.height = config['dataset_params']['height']

        self.classes = listdir(self.root)
        print(self.classes)
        self.classes.sort()
        self.files = []
        self.labels = []

        for i, c in enumerate(self.classes):
            new_files = [join(self.root, c,'%s.mat') %(f) for f in self.index]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = self.files[idx]
        data = loadmat(f)
        events = data['out1']
        events = np.array(events).astype(np.float32)
        events = events[:,[1,2,0,3]]
        events[events[:, -1] == 0, -1] = -1

        # events[:,2] = events[:,2]-np.min(events[:,2])
        # factor = 200
        # if np.max(events[:,2])>0:
        #     events[:,2] = events[:,2]/np.max(events[:,2]) * factor

        # if self.augmentation:
            # events = random_shift_events_new(events, max_shift=12,resolution=(self.height, self.width))
            # events = random_flip_events_along_x_new(events, resolution=(self.height, self.width))
            # events[:, 2] = (events[:, 2] + np.random.rand(1) * np.max(events[:,2])) % np.max(events[:,2])

        
        events_input=torch.from_numpy(events)
        
        flow_volume_output = (gen_discretized_event_volume(events = events_input, val_plus = False, events_val = events_input, vol_size = [6, self.pixel_size[0], self.pixel_size[1]]))
        flow_time_output = (per_event_timing_images(events_input, events_input, [2, self.pixel_size[0], self.pixel_size[1]], val_plus = False))
        flow_stacking_output = (per_stacking_events(events_input, events_input, [3, self.pixel_size[0], self.pixel_size[1]], val_plus = False))
        flow_counting_output = (per_event_counting_images(events_input, events_input, [2, self.pixel_size[0], self.pixel_size[1]], val_plus = False))
        flow_output = torch.cat([flow_volume_output, flow_time_output, flow_stacking_output, flow_counting_output], 0)
        
        count_vox = voxel_represent_count(events[:,0], events[:,1], events[:,2], 8, self.height, self.width)
        avg_vox = voxel_represent_avgt(events[:,0], events[:,1], events[:,2], 9, self.height, self.width)
        
        Range_Img = count_vox
        Residual_Img = residual_img(avg_vox)
        fus = torch.cat([Range_Img,Residual_Img]) 
        fus = torch.cat([fus, flow_output], 0)
        fus = normalize_event_volume(fus)
        
        events[events[:, -1] == -1, -1] = 0
        events = torch.from_numpy(events)
        
        assert not torch.any(torch.isnan(events))
        assert not torch.any(torch.isnan(fus))

        return fus.float(), events, label
    
    def sampler(self, events, nsample):
        # events number
        N = events.shape[0]
        if N>nsample:     
            # random select nsample index
            indices = torch.randperm(N)[:nsample]
            # get corresponding events
            sampled_events = events[indices]
        
        elif N<nsample:
            expend_factor = nsample//N
            sampled_events = events.repeat(expend_factor,1)
            num_copies = nsample%N
            indices_copy = torch.randperm(N)[:num_copies]
            sampled_events = torch.cat((sampled_events,events[indices_copy]),0)

        else:
            sampled_events = events

        return sampled_events
    
    
if __name__=='__main__':
    configs = FLAGS()    
    training_dataset = DVSCifar10(configs, mode='training')
    for i in range(0,1):
        training_dataset[i]
    # loader = Loader(dataset, 8, 2, True, 'cuda:0')

    # fuse, events, labels = dataset[109]

    # for fuse, events, labels in loader:
    #     print(fuse.shape, events.shape, labels)

    # data = loadmat('/cwang/home/gxs/Dataset/cifar10_dvs/airplane/0.mat')
    # print(data['out1'])