import argparse
from os.path import dirname
import torch
from datetime import datetime
import torchvision
import os
import collections
import yaml
import time
from easydict import EasyDict
import numpy as np
import tqdm
import sys
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter
from loss.Loss import cross_entropy_loss_and_accuracy
from data.loader_for_best import Loader
from data.cifar10_dvs import DVSCifar10
from model_ptv.get_model import get_model
import random #0 cal, 4 car,
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_seed(seed=4):
 
    random.seed(seed)
 
    np.random.seed(seed)
 
    torch.manual_seed(seed)

    torch.Generator().manual_seed(seed)
 
    torch.cuda.manual_seed(seed)
 
    torch.cuda.manual_seed_all(seed) 
 
    torch.backends.cudnn.benchmark = False
 
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.enabled = False

    os.environ['PYTHONHASHSEED'] = str(seed)

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def FLAGS():
    parser = argparse.ArgumentParser(
        """Deep Learning for Events. Supply a config file.""")

    # can be set in config
    parser.add_argument("--checkpoint", default='/home/ubuntu/lcl/cvpr2025/classification/output/DVSCifar10/model/49/model_best.pth')
    parser.add_argument("--test_dataset", default='/home/lxh/LCL/data/cifar10_dvs')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument('--config_path', default='/home/ubuntu/lcl/cvpr2025/classification/configs/config_DVSCifar10.yaml')
    parser.add_argument("--log_dir", default=os.path.join('/home/ubuntu/lcl/cvpr2025/classification/output/DVSCifar10/log', 
                                         str(0)))
    flags = parser.parse_args()
    config = load_yaml(flags.config_path)
    config.update(vars(flags))
    if os.path.exists(flags.log_dir) == False:
        os.makedirs(flags.log_dir, exist_ok=True)
    print(f"----------------------------\n"
          f"Starting testing with \n"
          f"checkpoint: {flags.checkpoint}\n"
          f"test_dataset: {flags.test_dataset}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"----------------------------")


    return EasyDict(config)


if __name__ == '__main__':
    configs = FLAGS()
    set_seed(4)
    torch.set_printoptions(precision=9)
    torch.set_printoptions(sci_mode=False)
    logfile = "{dir}/out.log".format(dir = configs.log_dir)
    errfile = "{dir}/err.log".format(dir = configs.log_dir)
    logfile = open(logfile, "a", 1)
    errfile = open(errfile, "a", 1)
    # sys.stdout = logfile
    # sys.stderr = errfile
    # test_dataset = NCaltech101(root=flags.test_dataset, augmentation=False)
    test_dataset = DVSCifar10(configs, mode='testing')


    # construct loader, responsible for streaming data to gpu
    test_loader = Loader(test_dataset, 
            batch_size = configs['dataset_params']['val_data_loader']['batch_size'], 
            num_workers = configs['dataset_params']['val_data_loader']['num_workers'], 
            pin_memory = True, 
            device = configs['dataset_params']['val_data_loader']['device'])


    # model, load and put to device
    # model = SalsaNextWithMotionAttention(nclasses=2, range_channel=8, n_input_scans=7, batch_size=flags.batch_size, height=240, width=240, num_batch=None, point_refine=None)
    from model_ptv.get_model import get_model

    model = get_model(configs['dataset_params']['num_classes'])

    ckpt = torch.load('/home/ubuntu/lcl/cvpr2025/classification/output/DVSCifar10/model/49/model_best.pth')
    state_dict = ckpt["state_dict"]
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # 去掉前缀（去掉前七个字符）
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    model.load_state_dict(new_state_dict, strict=True)  # 重新加载这个模型

    model = model.to(configs.device)

    criterion = cross_entropy_loss_and_accuracy
    model = model.eval()
    sum_accuracy = 0
    sum_loss = 0
    sum_time = 0
        
    correct = torch.zeros(configs['dataset_params']['num_classes']).cuda()
    total = torch.zeros(configs['dataset_params']['num_classes']).cuda()
    print("Test step")

    for fus, events, events_ptv3, coord, labels, grid_size, batch in tqdm.tqdm(test_loader):
        fus = fus.cuda()
        events = events.cuda()
        coord = coord.cuda()
        events_ptv3 = events_ptv3.cuda()
        labels = labels.cuda()
        batch = batch.cuda()
        grid_size = grid_size.cuda()
        dict_event = {'coord': coord,
            'feat': events_ptv3,
            'batch': batch,
            'fus': fus,
            'feat_ori': events,
            'grid_size': grid_size
            }
        with torch.no_grad():
            
            start_time = time.time()
            pred_labels = model(dict_event)
            end_time = time.time()
            sum_time += (end_time - start_time)
            
            prob = torch.nn.functional.softmax(pred_labels,dim=1)
            _, p = torch.max(prob, 1)
            loss, accuracy = criterion(pred_labels, labels.long())
        sum_accuracy += accuracy
        sum_loss += loss

        _, predicted = torch.max(pred_labels, 1)

        for j in range(configs['dataset_params']['num_classes']):
            correct[j] += (predicted[labels == j] == j).sum().item()
            total[j] += (labels == j).sum().item()

    print("Test time: ", sum_time / 1000)
        
    validation_loss = sum_loss.item() / len(test_loader)
    validation_accuracy = sum_accuracy.item() / len(test_loader)

    class_accuracies = correct / total
    class_accuracies[total == 0] = 0
    accuracy_output = ""
    for j in range(configs['dataset_params']['num_classes']):
        accuracy_output += f'Class {j}: {class_accuracies[j].item() * 100:.0f}% --- '
    print(accuracy_output[:-5])