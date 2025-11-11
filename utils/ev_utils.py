import torch
import numpy as np
from torchvision import utils as vutils
import os
import copy

#对事件做x,y轴方向分别随机位移
def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    events[:,0] += x_shift
    events[:,1] += y_shift

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]
    return events

#沿x轴对事件数据进行随机翻转
def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events

#对事件数据的时间戳进行反转
def Invert(events, p=0.5):
    max_t = np.max(events[:,2])
    if np.random.random() < p:
        events[:,2] = max_t - events[:,2]
    return events

#对事件数据的时间戳进行随机变换
def time_trans(events, p=0.5):
    if np.random.random() < p:
        a = np.random.rand()
        events[:,2] = (a*events[:,2]+1-a)*events[:,2]
    return events

def random_shift_events_new(events, max_shift=20, resolution=(180, 240), bounding_box=None):
    #随机平移
    H, W = resolution
    if bounding_box is not None:
        x_shift = np.random.randint(-min(bounding_box[0, 0], max_shift),
                                    min(W - bounding_box[2, 0], max_shift), size=(1,))
        y_shift = np.random.randint(-min(bounding_box[0, 1], max_shift),
                                    min(H - bounding_box[2, 1], max_shift), size=(1,))
        bounding_box[:, 0] += x_shift
        bounding_box[:, 1] += y_shift
    else:
        x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))

    events[:, 0] += x_shift
    events[:, 1] += y_shift


    valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
    events = events[valid_events]

    if bounding_box is None:
        return events

    return events, bounding_box

def random_flip_events_along_x_new(events, resolution=(180, 240), p=0.5, bounding_box=None):
    #左右翻转
    H, W = resolution
    flipped = False
    if np.random.random() < p:
        events[:, 0] = W - 1 - events[:, 0]
        flipped = True

    if bounding_box is None:
        return events

    if flipped:
        bounding_box[:, 0] = W - 1 - bounding_box[:, 0]
        bounding_box = bounding_box[[1, 0, 3, 2]]
    return events, bounding_box

def count_img_p(x, y, p):
    img = torch.sparse_coo_tensor([(p+1)/2, y, x], np.ones(p.shape[0]), size=(2, 180, 240)).to_dense()
    return img

def count_img(x, y):
    img = torch.sparse_coo_tensor(np.array([y, x]), np.ones(x.shape[0]), size=(180, 240)).to_dense()
    #####
    img = img/torch.max(img)
    return img

def global_count_img(x, y, t):
    ts = np.round((t-min(t))/(max(t)-min(t)) * 99)
    pos = np.zeros_like(t)
    t_dis = torch.sparse_coo_tensor([pos, ts], np.ones(t.shape[0]), size=(1, 100)).to_dense().squeeze()

    img = torch.sparse_coo_tensor([y, x], t_dis[ts], size=(180, 240)).to_dense()

    # print(img[img!=0])
    return img

def pos_img():
    posx = torch.zeros([180, 240], dtype=torch.float64)
    posy = torch.zeros([180, 240], dtype=torch.float64)
    for i in range(180):
        posx[i] = i
    for j in range(240):
        posy[:, j] = j
    return posx, posy

def newt_img(x, y, t):
    ts = torch.tensor((t-min(t))/(max(t)-min(t)), dtype=torch.float64)
    newt = torch.zeros([180, 240], dtype=torch.float64)
    ind = 240 * y + x
    ind = torch.tensor(ind/1.0, dtype=torch.long)
    newt.view(-1).put_(ind, ts, accumulate=False)  
    return newt

def avgt_img(x, y, t):
    ts = (t-min(t))/(max(t)-min(t))
    count = torch.sparse_coo_tensor(np.array([y, x]), np.ones(x.shape[0]), size=(180, 240)).to_dense()
    avg = torch.sparse_coo_tensor(np.array([y, x]), ts, size=(180, 240)).to_dense()
    avg[count!=0] /= count[count!=0]
    ########
    avg = avg/torch.max(avg)
    return avg

def vart_img(x, y, t):
    ts = (t-min(t))/(max(t)-min(t))
    count = torch.sparse_coo_tensor(np.array([y, x]), np.ones(x.shape[0]), size=(180, 240)).to_dense()
    avg = torch.sparse_coo_tensor(np.array([y, x]), ts, size=(180, 240)).to_dense()
    avg[count!=0] /= count[count!=0]
    ts = (torch.tensor(ts) - avg[y, x])**2
    var = torch.sparse_coo_tensor(np.array([y, x]), ts, size=(180, 240)).to_dense()
    var[count!=0] /= count[count!=0]
    ########
    var = var/torch.max(var)
    return var

def xt_yt(x, y, t):
    scale_t = np.random.randint(1, 200, 1)
    # print(scale_t)
    ts = np.round((t-min(t))/(max(t)-min(t)) * scale_t)
    t_max = int(max(ts))
    xt = torch.sparse_coo_tensor([ts, x], np.ones(x.shape[0]), size=(t_max+1, 240)).to_dense()
    yt = torch.sparse_coo_tensor([ts, y], np.ones(x.shape[0]), size=(t_max+1, 180)).to_dense()
    return xt.unsqueeze(0), yt.unsqueeze(0)

def make_frames(x, y, t, p, max_count):
    count = torch.zeros([180, 240])
    pos = torch.tensor([x, y], dtype=torch.int32).T
    frames = []
    num = 0
    if os.path.exists('./s') == False:
        os.makedirs('./s', exist_ok=True)
    for i, c in enumerate(pos):
        num += 1
        if count[c[1], c[0]] <= max_count:      
            count[c[1], c[0]] += 1
        else:
            count[count!=0] = 1
            if frames == []:
                frames = count.unsqueeze(0)
            else:
                frames = torch.cat((frames, count.unsqueeze(0)), dim=0)
            vutils.save_image(count, './s/%d.jpg'%(i))
            count[:] = 0
            num = 0
    return frames

def see(x, y, t):
    ts = np.round((t-min(t))/(max(t)-min(t)) * 99)
    pos = np.zeros_like(t)
    t_dis = torch.sparse_coo_tensor([pos, ts], np.ones(t.shape[0]), size=(1, 100)).to_dense()
    # x_dis = torch.sparse_coo_tensor([pos, np.round((x-min(x))/(max(x)-min(x)) * 99)], np.ones(x.shape[0]), size=(1, 100)).to_dense()
    # y_dis = torch.sparse_coo_tensor([pos, np.round((y-min(y))/(max(y)-min(y)) * 99)], np.ones(x.shape[0]), size=(1, 100)).to_dense()
    # print(t_dis)
    # print(t_dis.shape)
    # plt.plot(np.arange(0,100,1), t_dis[0])
    # plt.plot(np.arange(0,100,1), x_dis[0])
    # plt.plot(np.arange(0,100,1), y_dis[0])
    # plt.savefig('./see.jpg')
    return t_dis.squeeze()

def voxel_represent_count(x, y, t, bin_num, h=180, w=240):
    ts = np.round((t-min(t))/(max(t)-min(t)) * (bin_num-1))
    count = torch.sparse_coo_tensor(np.array([ts, y.tolist(), x.tolist()]), np.ones(x.shape[0]), size=(bin_num, h, w)).to_dense()
    count = count/torch.max(count)
    # for i in range(bin_num):
    #     if torch.max(count[i]):
    #         count[i] = count[i]/torch.max(count[i])
    return count

# 分为bin_num个切片，每个切片分别包含平均时间值
def voxel_represent_avgt(x, y, t, bin_num, h=180, w=240):
    value_t = (t-min(t))/(max(t)-min(t))
    ts = np.round(value_t * (bin_num-1)).tolist()
    for i in range(bin_num):
        if value_t[ts==i].shape[0]>0 and (np.max(value_t[ts==i])-np.min(value_t[ts==i]))>0:
            value_t[ts==i] = (value_t[ts==i]-np.min(value_t[ts==i])) / (np.max(value_t[ts==i])-np.min(value_t[ts==i]))
    count = torch.sparse_coo_tensor(np.array([ts, y.tolist(), x.tolist()]), np.ones(x.shape[0]), size=(bin_num, h, w)).to_dense() 
    # torch.sparse_coo_tensor用法：[ts, y.tolist(), x.tolist()]为索引的三维坐标，例如：将(ts[0],y[0],x[0])位置赋值为np.ones(x.shape[0][0])
    avg = torch.sparse_coo_tensor(np.array([ts, y.tolist(), x.tolist()]), value_t, size=(bin_num, h, w)).to_dense()
    avg[count!=0] /= count[count!=0]
    return avg

def residual_img(img):
    bin_num, h, w = img.shape
    residual = torch.zeros([bin_num-1, h, w], dtype=torch.float)
    for i in range(bin_num-1):
        residual[i] = torch.abs(img[bin_num-1] - img[i])
        if torch.max(residual[i])!=0:
            residual[i] /= torch.max(residual[i])
    return residual

def residual_raw_t(x,y,t,bin_num):
    value_t = (t-min(t))/(max(t)-min(t))
    ts = np.round(value_t * (bin_num-1)) 
    count = torch.sparse_coo_tensor(np.array([ts, y, x]), np.ones(x.shape[0]), size=(bin_num, 180, 240)).to_dense()
    avg = torch.sparse_coo_tensor(np.array([ts, y, x]), t, size=(bin_num, 180, 240)).to_dense()
    avg[count!=0] /= count[count!=0]
    residual = torch.zeros([bin_num-1, 180, 240], dtype=torch.float)
    for i in range(bin_num-1):
        residual[i] = avg[i+1]-avg[i]
    # #########
    residual = torch.cat((avg[0].unsqueeze(0), residual), dim=0)
    # residual = torch.abs(residual)
    residual[residual<0]=0
    residual /= torch.max(residual)
    # for i in range(res.shape[0]):
        #     img = res[i]
        #     img[img!=0]=1
        #     vutils.save_image(img, './img%d.jpg'%(i))
    return residual

def residual_img_2c(img):
    bin_num = img.shape[0]
    residual = torch.zeros([2*(bin_num-1), 180, 240], dtype=torch.float)
    for i in range(bin_num-1):
        temp = img[bin_num-1] - img[i]
        residual[2*i, temp>0] = temp[temp>0]
        residual[2*i+1, temp<0] = torch.abs(temp[temp<0])
        if torch.max(residual[2*i]):
            residual[2*i] /= torch.max(residual[2*i:2*i+2])
        if torch.max(residual[2*i+1]):
            residual[2*i+1] /= torch.max(residual[2*i:2*i+2])
    return residual

def residual_img_c(img):
    bin_num = img.shape[0]
    count_vox = copy.deepcopy(img)
    count_vox[count_vox!=0] = 1
    residual = torch.zeros([2*(bin_num-1), 180, 240], dtype=torch.float)
    for i in range(bin_num-1):
        temp_subtrack = img[bin_num-1] - img[i]
        temp_mul = img[bin_num-1] * img[i]
        residual[i*2, temp_subtrack==1] = 1
        residual[i*2+1, temp_subtrack==-1] = 1
        residual[i*2:i*2+2, temp_mul==1] = 0.5
    return residual

def residual_img_t(img):
    bin_num = img.shape[0]
    res = copy.deepcopy(img)
    for i in range(bin_num):
        res[i] -= torch.mean(res[i])
        res[i, res[i]<0] = 0
        res[i] = (res[i] - torch.min(res[i])) / (torch.max(res[i]) - torch.min(res[i]))
    return res

def Fxy(x, y, t, p, B, plane=[240,180]):
    t_norm = (B-1)*(t-min(t))/(max(t)-min(t))
    frame = []
    for i in range(B):
        value = p*(1-np.abs(i-t_norm))
        img = torch.sparse_coo_tensor(np.array([y, x]), value, size=(plane[1], plane[0])).to_dense()
        frame.append(img)
    frame = torch.stack(frame)
    return frame

def Fxt(x, y, t, p, B, T=240, plane=[240,240]):
    t_norm = np.floor((T-1)*(t-min(t))/(max(t)-min(t)))
    y_norm = (B-1)*y/180
    frame = []
    for i in range(B):
        value = p*(1-np.abs(i-y_norm))
        img = torch.sparse_coo_tensor(np.array([t_norm, x]), value, size=(plane[1], plane[0])).to_dense()
        frame.append(img)
    frame = torch.stack(frame)
    return frame

def Fyt(x, y, t, p, B, T=180, plane=[180,180]):
    t_norm = np.floor((T-1)*(t-min(t))/(max(t)-min(t)))
    x_norm = (B-1)*x/240
    frame = []
    for i in range(B):
        value = p*(1-np.abs(i-x_norm))
        img = torch.sparse_coo_tensor(np.array([t_norm, y]), value, size=(plane[1], plane[0])).to_dense()
        frame.append(img)
    frame = torch.stack(frame)
    return frame