import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

def random_sample_events(events, k):
    events = events.transpose(1,0)
    if events.shape[1] >= k:
        choice_idx = np.random.choice(events.shape[1], k, replace=False)
    else:
        fix_idx = np.asarray(range(events.shape[1]))
        while events.shape[1] + fix_idx.shape[0] < k:
            fix_idx = np.concatenate((fix_idx, np.asarray(range(events.shape[1]))), axis=0)
        random_idx = np.random.choice(events.shape[1], k - fix_idx.shape[0], replace=False)
        choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
    events = events[:, choice_idx]
    return events.transpose(1,0)

class Loader:
    def __init__(self, dataset, batch_size, num_workers, pin_memory, device):
        self.device = device
        split_indices = list(range(len(dataset)))
        sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
        # sampler = torch.utils.data.SequentialSampler(split_indices)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                             num_workers=num_workers, pin_memory=pin_memory,
                                             collate_fn=collate_events_caltech101, 
                                             drop_last=True)

    def __iter__(self):
        for data in self.loader:
            data = [d for d in data]
            yield data

    def __len__(self):
        return len(self.loader)
    
class Loader_for_N_imagenet:
    def __init__(self, dataset, batch_size, num_workers, pin_memory, device):
        self.device = device
        split_indices = list(range(len(dataset)))
        sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
        # sampler = torch.utils.data.SequentialSampler(split_indices)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                             num_workers=num_workers, pin_memory=pin_memory,
                                             collate_fn=collate_events_caltech101, 
                                             drop_last=True)

    def __iter__(self):
        for data in self.loader:
            data = [d for d in data]
            yield data

    def __len__(self):
        return len(self.loader)


class RasEventCloud:
    def __init__(self, input_size: tuple):
        assert len(input_size) == 3
        self.channels = input_size[0]
        self.height = input_size[1]
        self.width = input_size[2]
        self.e_cloud_list = list()
        self.cloud_channel_grid = torch.ones(input_size, dtype=torch.float, requires_grad=False)

        for i in range(self.channels):
            self.cloud_channel_grid[i] *= i

        y_cord = torch.tensor([i for i in range(self.height)])
        x_cord = torch.tensor([i for i in range(self.width)])
        y, x = torch.meshgrid(y_cord, x_cord, indexing='ij')
        y = y.unsqueeze(0)
        x = x.unsqueeze(0)
        self.y_grid = y.expand(self.channels, -1, -1)
        self.x_grid = x.expand(self.channels, -1, -1)

    def convert(self, events, num_sample = 1024):
        """
        return data: channels, x, y, t_avg, p_acc, e_count
        """
        C, H, W = self.channels, self.height, self.width
        cloud_polar_grid = torch.zeros((C, H, W), dtype=torch.float, requires_grad=False)
        cloud_time_acc_grid = torch.zeros((C, H, W), dtype=torch.float, requires_grad=False)
        cloud_event_count_grid = torch.zeros((C, H, W), dtype=torch.float, requires_grad=False)
        # 将时间放缩到[0,channels]
        events[:, 2] = events[:, 2] - events[:, 2].min()
        events[:, 2] = (events[:, 2] / (events[:, 2].max())) / 1.001
        events[:, 2] = events[:, 2] * self.channels
        with torch.no_grad():
            t = events[:, 2]
            p = 2 * events[:, 3] - 1
            t0 = events[:, 2].int()
            x = events[:, 0].int() - 1
            y = events[:, 1].int() - 1

            index = H * W * t0.long() + W * y.long() + x.long()
            cloud_polar_grid.put_(index, p.float(), accumulate=True) #根据index进行赋值
            cloud_time_acc_grid.put_(index, (t - t0).float(), accumulate=True) #
            cloud_event_count_grid.put_(index, torch.ones_like(x, dtype=torch.float), accumulate=True) #计数操作
            valid_mask = torch.ne(cloud_event_count_grid, 0)
            data_list = list()
            data_list.append(self.cloud_channel_grid[valid_mask].reshape(-1, 1))
            data_list.append(self.x_grid[valid_mask].reshape(-1, 1))
            data_list.append(self.y_grid[valid_mask].reshape(-1, 1))
            data_list.append(cloud_time_acc_grid[valid_mask].reshape(-1, 1))
            data_list.append(cloud_polar_grid[valid_mask].reshape(-1, 1))
            data_list.append(cloud_event_count_grid[valid_mask].reshape(-1, 1))
            #channels, x, y, t_avg, p_acc, e_count

            data = torch.cat(data_list, 1)
            data[:, 3] /= data[:, 5]#平均时间（小块内的）
            data[:, 3] = data[:, 3] / C + data[:, 0] / C 
            return data #(n,6)
        
def random_sample_point(xyz, npoint):
    """
    
    输入:
        xyz: 点云数据张量, [N, C], C => [x, y, t, p]
        npoint: 采样后的点数量
        N > npoint 或 N < npoint 或 N == npoint
    返回:
        xyz_sample: 采样后的点云数据, [npoint, C]
        centroids: 采样点的索引, [npoint]
    """
    N, C = xyz.shape

    if npoint <= N:
        # 如果需要采样的点数少于或等于原始点数，从中选择npoint个点
        IndexSelect = torch.randperm(N)[:npoint].sort().values
    else:
        # 如果需要采样的点数多于原始点数，允许重复选择点
        IndexSelect = torch.randint(0, N, (npoint,)).sort().values

    xyz_sample = xyz[IndexSelect, :]

    return xyz_sample, IndexSelect
       
def RasEventCloud_preprocess_cifar10(data, num_sample = 0, k = 16):

    if num_sample == 0:
        EventCloudDHP = RasEventCloud(input_size=(k, 128, 128))
        data = EventCloudDHP.convert(data).numpy()[:, 1:]
        return data
    data = data[:, 0:4]
    EventCloudDHP = RasEventCloud(input_size=(k, 128, 128)) #1 处理手工特征 预处理
    data = EventCloudDHP.convert(data, num_sample)[:, 1:]  # [x, y, t_avg, p_acc, event_cnt]
    data = data.numpy()
    data[:,2] = data[:,2] * 128
    
    if num_sample != 0:
        data_sample, _ = random_sample_point(data, num_sample)# 2 采样num_sample个点
        data = data_sample  # [num_sample, C]
    return data

def RasEventCloud_preprocess_Caltech101(data, num_sample = 0, k = 16):

    if num_sample == 0:
        EventCloudDHP = RasEventCloud(input_size=(k, 180, 240))
        data = EventCloudDHP.convert(data).numpy()[:, 1:]
        return data
    data = data[:, 0:4]
    EventCloudDHP = RasEventCloud(input_size=(k, 180, 240))
    data = EventCloudDHP.convert(data, num_sample)[:, 1:]  # [x, y, t_avg, p_acc, event_cnt]
    data = data.numpy()
    data[:,2] = data[:,2] * 128
    
    if num_sample != 0:
        data_sample, _ = random_sample_point(data, num_sample)
        data = data_sample  # [num_sample, C]
    return data

def RasEventCloud_preprocess_imagenet(data, num_sample = 0, k = 16):

    if num_sample == 0:
        EventCloudDHP = RasEventCloud(input_size=(k, 480, 640))
        data = EventCloudDHP.convert(data).numpy()[:, 1:]
        return data
    data = data[:, 0:4]
    EventCloudDHP = RasEventCloud(input_size=(k, 480, 640))
    data = EventCloudDHP.convert(data, num_sample)[:, 1:]  # [x, y, t_avg, p_acc, event_cnt]
    data = data.numpy()
    data[:,2] = data[:,2] * 128
    
    if num_sample != 0:
        data_sample, _ = random_sample_point(data, num_sample)
        data = data_sample  # [num_sample, C]
    return data
    

def normalize_channel(flow, channel_idx):
        # 计算通道的最大值和最小值
        input_max, input_min = flow[:, :, channel_idx].max(dim=1, keepdim=True)[0], flow[:, :, channel_idx].min(dim=1, keepdim=True)[0]
        # 归一化处理
        for i in range(flow.shape[0]):
            if torch.equal(input_max[i], input_min[i]):
                flow[i, :, channel_idx] = 0  # 用 0 填充
            else:
                flow[i, :, channel_idx] = (flow[i, :, channel_idx] - input_min[i]) / (input_max[i] - input_min[i])
        return flow

def norm(flow, pixel_size):
    # 将坐标归一化
    flow[:, :, 0] = flow[:, :, 0] / (pixel_size[0] - 1)
    flow[:, :, 1] = flow[:, :, 1] / (pixel_size[1] - 1)
    
    # 批量处理第 2、3、4 个通道
    for channel_idx in [2, 3, 4]:
        flow = normalize_channel(flow, channel_idx)
    
    return flow

def collate_events_cifar10(data):
    fus = []
    labels = []
    events = []
    events_ptv3s = []
    batch_indices = []
    for i, d in enumerate(data):
        labels.append(d[2])
        image = F.interpolate(d[0].unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
        fus.append(image)
        flow = d[1]
        event = RasEventCloud_preprocess_cifar10(flow, 4096, 8)
        events.append(event)
        batch_index = torch.full((event.shape[0],), i, dtype=torch.long)
        batch_indices.append(batch_index)
    batch = [torch.full((d.shape[0],), i) for i, d in enumerate(events)]
    batch = torch.cat(batch).long()
    events = default_collate(events)
    events = norm(events,[128,128])
    labels = default_collate(labels)
    fus = default_collate(fus)
    events_ptv3s = events.view(-1,5)
    coords = events_ptv3s[:,0:3]
    grid_size = torch.tensor([0.01])
    return fus, events, events_ptv3s, coords, labels, grid_size, batch

def collate_events_N_imagenet(data):
    fus = []
    labels = []
    events = []
    events_ptv3s = []
    batch_indices = []
    for i, d in enumerate(data):
        labels.append(d[2])
        fus.append(d[0])
        flow = d[1]
        event = RasEventCloud_preprocess_imagenet(flow, 8192, 16)
        events.append(event)
        batch_index = torch.full((event.shape[0],), i, dtype=torch.long)
        batch_indices.append(batch_index)
    batch = [torch.full((d.shape[0],), i) for i, d in enumerate(events)]
    batch = torch.cat(batch).long()
    sliced_batch = batch[0 : batch.shape[0] // 3]  # 形状 [8192]
    batchs = sliced_batch.unsqueeze(0).repeat(3, 1)  # 形状 [3, 8192]
    events = default_collate(events)
    events = norm(events,[480,640])
    labels = default_collate(labels)
    fus = default_collate(fus)
    events_ptv3s = events.view(-1,5)
    coords = events_ptv3s[:,0:3]
    grid_size = torch.tensor([0.01])
    return fus, events, events_ptv3s, coords, labels, grid_size, batchs

def collate_events_caltech101(data):
    fus = []
    labels = []
    events = []
    events_ptv3s = []
    batch_indices = []
    for i, d in enumerate(data):
        labels.append(d[2])
        image = F.interpolate(d[0].unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
        fus.append(image)
        flow = d[1]
        event = RasEventCloud_preprocess_Caltech101(flow, 4096, 8)
        # event = RasEventCloud_preprocess_cifar10(flow, 4096, 16)
        events.append(event)
        batch_index = torch.full((event.shape[0],), i, dtype=torch.long)
        batch_indices.append(batch_index)
    batch = [torch.full((d.shape[0],), i) for i, d in enumerate(events)]
    batch = torch.cat(batch).long()
    events = default_collate(events)
    events = norm(events,[240,180])
    # events[:,:,2] = events[:,:,2] * 2
    labels = default_collate(labels)
    fus = default_collate(fus)
    events_ptv3s = events.view(-1,5)
    coords = events_ptv3s[:,0:3]
    grid_size = torch.tensor([0.01])
    return fus, events, events_ptv3s, coords, labels, grid_size, batch

def collate_events_N_imagenet2222222(data):
    fus = []
    labels = []
    events = []
    events_ptv3s = []
    batch_indices = []
    for i, d in enumerate(data):
        labels.append(d[2])
        image = F.interpolate(d[0].unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
        fus.append(image)
        flow = d[1]
        event = RasEventCloud_preprocess_imagenet(flow, 8192, 16)
        events.append(event)
        batch_index = torch.full((event.shape[0],), i, dtype=torch.long)
        batch_indices.append(batch_index)
    batch = [torch.full((d.shape[0],), i) for i, d in enumerate(events)]
    batch = torch.cat(batch).long()
    sliced_batch = batch[0 : batch.shape[0] // 3]  # 形状 [8192]
    # batchs = sliced_batch.unsqueeze(0).repeat(3, 1)  # 形状 [3, 8192]
    events = default_collate(events)
    events = norm(events,[480,640])
    labels = default_collate(labels)
    fus = default_collate(fus)
    events_ptv3s = events.view(-1,5)
    coords = events_ptv3s[:,0:3]
    grid_size = torch.tensor([0.01])
    return fus, events, events_ptv3s, coords, labels, grid_size, batch
