import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from model_ptv.utils_ep2t import *
from PointTransformerV3.model_ptv3 import PointTransformerV3

class Attention(nn.Module):
    def __init__(self, channels, num_heads = 1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=True)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=True)
        self.v_conv1 = nn.Conv1d(channels, channels, 1, bias=True)
        self.v_conv2 = nn.Conv1d(channels, channels, 1, bias=True)
        self.fcs_t1 = nn.Linear(2 * channels, 2 * channels, bias=True)
        self.fcs_t2 = nn.Linear(2 * channels, channels, bias=True)

        self.fcs_s1 = nn.Linear(2 * channels, 2 * channels, bias=True)
        self.fcs_s2 = nn.Linear(2 * channels, channels, bias=True)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, y):
        batch_size = x.size(0)

        x_q = self.q_conv(y).view(batch_size, self.num_heads, self.head_dim, -1)
        x_k = self.k_conv(x).view(batch_size, self.num_heads, self.head_dim, -1)
        x_v = self.v_conv1(x).view(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        y_v = self.v_conv2(y).view(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        
        x_con = torch.cat([x_q, x_k], dim=2)
        v_con = torch.cat([y_v, x_v], dim=3)
        energy = torch.matmul(x_con, x_con.permute(0, 1, 3, 2)) / ((2 * self.head_dim) ** 0.5)
        energy = self.act(self.fcs_t1(energy))
        energy = self.act(self.fcs_t2(energy))

        attention = self.softmax(energy)
        attention = self.dropout(attention)


        x_r = torch.matmul(v_con, attention) # b, c, n 
        x_r = x_r.permute(0, 1, 3, 2).contiguous().view(batch_size, -1, x.size(2))
        
        return x_r

class STA(nn.Module):
    def __init__(self, in_channel):
        super(STA, self).__init__()
        self.time_cor_1 = Attention(in_channel)
        self.space_cor_1 = Attention(in_channel)

        self.time_cor_2 = Attention(in_channel)
        self.space_cor_2 = Attention(in_channel)

        self.time_cor_3 = Attention(in_channel)
        self.space_cor_3 = Attention(in_channel)

        self.time_cor_4 = Attention(in_channel)
        self.space_cor_4 = Attention(in_channel)

        self.time_cor_5 = Attention(in_channel)
        self.space_cor_5 = Attention(in_channel)

        self.time_cor_6 = Attention(in_channel)
        self.space_cor_6 = Attention(in_channel)

        self.fcs1 = nn.Linear(2 * in_channel, 4 * in_channel)
        self.fcs2 = nn.Linear(4 * in_channel, 2 * in_channel)

        self.norm_time_1 = nn.LayerNorm([in_channel, 4096])
        self.norm_space_1 = nn.LayerNorm([in_channel, 4096])

        self.norm_time_2 = nn.LayerNorm([2 * in_channel, 4096])
        self.norm_space_2 = nn.LayerNorm([2 * in_channel, 4096])

        self.output = nn.Conv1d(in_channel, in_channel, 1)
        self.sig = torch.nn.Sigmoid()
        self.act = nn.ReLU()  # ReLU activation function

    def forward(self, x): 
        B, C, N = x[:, 2].shape
        global_x = x[:, 2]
        time_x = x[:, 0]
        space_x = x[:, 1]
        
        # Residual connections added after each attention operation
        time_residual = time_x
        space_residual = space_x
        
        # First set of attention layers
        time_plus = self.time_cor_1(space_x, time_x)
        space_plus = self.space_cor_1(time_x, space_x)
        time_x = self.act(time_plus) + time_residual  # Add residual connection
        space_x = self.act(space_plus) + space_residual  # Add residual connection
        time_x = self.norm_time_1(time_x)
        space_x = self.norm_space_1(space_x)

        time_residual = time_x  # Update residual for next layer
        space_residual = space_x  # Update residual for next layer

        # Second set of attention layers
        time_plus = self.time_cor_2(space_x, time_x)
        space_plus = self.space_cor_2(time_x, space_x)
        time_x = self.act(time_plus) + time_residual  # Add residual connection
        space_x = self.act(space_plus) + space_residual  # Add residual connection
        time_x = self.norm_time_1(time_x)
        space_x = self.norm_space_1(space_x)

        time_residual = time_x  # Update residual for next layer
        space_residual = space_x  # Update residual for next layer

        # Third set of attention layers
        time_plus = self.time_cor_3(space_x, time_x)
        space_plus = self.space_cor_3(time_x, space_x)
        time_x = self.act(time_plus) + time_residual  # Add residual connection
        space_x = self.act(space_plus) + space_residual  # Add residual connection
        time_x = self.norm_time_1(time_x)
        space_x = self.norm_space_1(space_x)

        time_residual = time_x  # Update residual for next layer
        space_residual = space_x  # Update residual for next layer

        # Fourth set of attention layers
        time_plus = self.time_cor_4(space_x, time_x)
        space_plus = self.space_cor_4(time_x, space_x)
        time_x = self.act(time_plus) + time_residual  # Add residual connection
        space_x = self.act(space_plus) + space_residual  # Add residual connection
        time_x = self.norm_time_1(time_x)
        space_x = self.norm_space_1(space_x)

        time_residual = time_x  # Update residual for next layer
        space_residual = space_x  # Update residual for next layer
        
        # Global correlation with updated residuals
        global_t_plus = self.time_cor_5(global_x, time_x)
        global_s_plus = self.space_cor_5(global_x, space_x)

        global_t_plus = self.act(global_t_plus) + time_residual  # Add residual connection
        global_s_plus = self.act(global_s_plus) + space_residual  # Add residual connection
        global_t_plus = self.norm_time_1(global_t_plus)
        global_s_plus = self.norm_space_1(global_s_plus)

        time_residual = global_t_plus  # Update residual for next layer
        space_residual = global_s_plus  # Update residual for next layer

        global_t_plus = self.time_cor_6(global_s_plus, time_x)
        global_s_plus = self.space_cor_6(global_t_plus, space_x)

        global_t_plus = self.act(global_t_plus) + time_residual  # Add residual connection
        global_s_plus = self.act(global_s_plus) + space_residual  # Add residual connection
        global_t_plus = self.norm_time_1(global_t_plus)
        global_s_plus = self.norm_space_1(global_s_plus)

        global_x = torch.cat([global_t_plus, global_s_plus], dim=1).view(B,N,-1)
        global_x = self.act(self.fcs1(global_x))
        global_x = self.act(self.fcs2(global_x))
        global_x = global_x.view(B,-1,N)
        global_x = self.norm_time_2(global_x).view(B,N,-1)

        return global_x

    
class PointNetEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(PointNetEncoder, self).__init__()
        self.stage_1 = PointTransformerV3(enc_channels=(64, 128, 256),enc_num_head=(4, 8, 16),enc_patch_size=(512,512,512),
                                          dec_channels=(64, 128),dec_num_head=(4, 8),dec_patch_size=(512,512))
        
        self.stage_2 = PointTransformerV3(enc_channels=(64, 128, 256),enc_num_head=(4, 8, 16),enc_patch_size=(512,512,512),
                                          dec_channels=(64, 128),dec_num_head=(4, 8),dec_patch_size=(512,512))
        
        self.stage_3 = PointTransformerV3(order=["z", "z-trans", "hilbert", "hilbert-trans"],stride=(2, 2, 2, 2),
                                          enc_depths=(2, 2, 2, 6, 2),enc_channels=(32, 64, 128, 256, 512),
                                          enc_num_head=(2, 4, 8, 16, 32),enc_patch_size=(512,512,512,512,512),
                                          dec_depths=(2, 2, 2, 2),dec_channels=(64, 64, 128, 256),
                                          dec_num_head=(4, 4, 8, 16),dec_patch_size=(512,512,512,512))
        self.STA_module = STA(64)
        self.output = nn.Conv1d(2 * embedding_size, embedding_size, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.act = nn.GELU()
        self.output2 = nn.Conv1d(embedding_size, 64, 1)
        weight_list = [[0.0, 1.0], [1.0, 0.0]]
        weight_tensor = torch.tensor(weight_list)
        self.trainable_weight = torch.nn.Parameter(weight_tensor)
    def compute_distance_matrix(self, x1, x2):
        """
        计算两个点集之间的距离矩阵。

        参数:
        - x1: 第一个点集，大小为 [B, N1, 3]
        - x2: 第二个点集，大小为 [B, N2, 3]

        返回:
        - dist: 距离矩阵，大小为 [B, N1, N2]    
        """
        # 计算点集之间的差值
        diff = x1.unsqueeze(2) - x2.unsqueeze(1)
        # 计算欧氏距离的平方
        dist_sq = torch.sum(diff ** 2, dim=-1)

        # 开方得到距离矩阵
        dist = torch.sqrt(dist_sq)

        return dist
    
    def compute_distance_matrix_weight(self, src, dst, weight):
        """
        计算两个点集之间的距离矩阵。

        参数:
        - x1: 第一个点集，大小为 [B, N1, 3]
        - x2: 第二个点集，大小为 [B, N2, 3]

        返回:
        - dist: 距离矩阵，大小为 [B, N1, N2]    
        """
        res = []
        B, N, _ = src.shape
        _, M, _ = dst.shape
        for i in range(len(weight)):
            src_dist = src[:, :, :2]
            dst_dist = dst[:, :, :2]
            dist_xy = -2 * torch.matmul(src_dist, dst_dist.permute(0, 2, 1))
            dist_xy += torch.sum(src_dist ** 2, -1).view(B, N, 1)
            dist_xy += torch.sum(dst_dist ** 2, -1).view(B, 1, M)
            dist_xy = torch.clamp(dist_xy, min=1e-5)

            src_dist = src[:, :, 2].unsqueeze(-1)
            dst_dist = dst[:, :, 2].unsqueeze(-1)
            dist_t = -2 * torch.matmul(src_dist, dst_dist.permute(0, 2, 1))
            dist_t += torch.sum(src_dist ** 2, -1).view(B, N, 1)
            dist_t += torch.sum(dst_dist ** 2, -1).view(B, 1, M)
            dist_t = torch.clamp(dist_t, min=1e-5)

            dist = torch.sqrt(dist_xy) * weight[i][0] + torch.sqrt(dist_t) * weight[i][1] 
            res.append(dist)
        res = torch.stack(res, 1)
        
        return res
    
    def forward(self, dict_event):
        feat = dict_event['feat_ori']
        B, N, _ = feat.shape
        l1_point = []
        coord_ori = dict_event['coord']

        coord_t = copy.deepcopy(dict_event['coord'])
        coord_t[:,0:2] = 0


        coord_s = copy.deepcopy(dict_event['coord'])
        coord_s[:,2] = 0
        
        dict_event['coord'] = coord_t
        t_out = self.stage_1(dict_event)
        t_out = t_out['feat'].view(B,N,-1).permute(0, 2, 1)

        dict_event['coord'] = coord_s
        s_out = self.stage_2(dict_event)  
        s_out = s_out['feat'].view(B,N,-1).permute(0, 2, 1)

        dict_event['coord'] = coord_ori
        ptv_out = self.stage_3(dict_event)
        ptv_out = ptv_out['feat'].view(B,N,-1).permute(0, 2, 1)
        # print(ptv_out[0,0:10,:])

        l1_point.append(t_out)
        l1_point.append(s_out)
        l1_point.append(ptv_out)
        l1_points = torch.stack(l1_point, 1)
        l1_points = self.STA_module(l1_points).view(B, -1, N)
        x = self.output(l1_points)
        return ptv_out
            
class PointNetAutoencoder(nn.Module):
    def __init__(
        self, embedding_size, input_channels=4, feat_compute = False, output_channels=4, normalize=True
    ):
        super(PointNetAutoencoder, self).__init__()
        self.normalize = normalize
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embedding_size = embedding_size
        self.encoder = PointNetEncoder(embedding_size)
        
        
    def forward(self, dict_event):
        z = self.encode(dict_event)
        return z

    def encode(self, dict_event):
        z = self.encoder(dict_event)
        
        
        if self.normalize:
            z = F.normalize(z)
        return z
