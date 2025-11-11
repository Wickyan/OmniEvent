import torch
import torch.nn as nn
from model_ptv.dela_for_best import PointNetAutoencoder
from model_ptv.event_utils import *

class EP2T(nn.Module):
    def __init__(self, pixel_size=(256, 256), embedding_size=256):
        super(EP2T, self).__init__()
        self.pixel_size = pixel_size
        self.flow_encoder = PointNetAutoencoder(embedding_size)
    def forward(self, dict_event):
        flow_ori = dict_event['feat_ori']
        flow_res = self.flow_encoder(dict_event) # 到这里还是点的特征，还没有Tensor化,得到的为[B,C,N]
        flow_feature_output = (gen_feature_dense_event_3(events = flow_ori, events_val = flow_res.transpose(-2,-1), \
                                vol_size = [self.pixel_size[0], self.pixel_size[1]]))    
                                # 该函数用于将输入的事件张量化表示,输入为[B,N,C],输出为[B,C,H,W]
        return flow_feature_output

class EP2T2(nn.Module):
    def __init__(self, pixel_size=(256, 256), embedding_size=256):
        super(EP2T2, self).__init__()
        self.pixel_size = pixel_size
        self.flow_encoder = PointNetAutoencoder(embedding_size)
    def forward(self, dict_event):
        flow_ori = dict_event['feat_ori']
        flow_res = self.flow_encoder(dict_event) # 到这里还是点的特征，还没有Tensor化,得到的为[B,C,N]
        # print(flow_res.shape)
        # print(flow_ori.shape)
        # print(dict_event['batch'].shape)
        flow_feature_output = (gen_feature_dense_event_4(events_ori = flow_ori, events_ptv3 = flow_res, \
                                vol_size = [self.pixel_size[0], self.pixel_size[1]], batch = dict_event['batch']))    
                                # 该函数用于将输入的事件张量化表示,输入为[B,N,C],输出为[B,C,H,W]
        return flow_feature_output