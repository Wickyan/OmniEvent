import argparse
from os.path import dirname
import torch
import os
import yaml
from easydict import EasyDict
import collections
import numpy as np
import tqdm
import sys
import shutil
import glob
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter
from loss.Loss import cross_entropy_loss_and_accuracy
from data.loader_aaai import Loader
from data.cifar10_dvs import DVSCifar10
from model_ptv.get_model import get_model
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

def find_next_folder_index(parent_dir):
    if not os.path.exists(parent_dir) or not os.path.isdir(parent_dir):
        return 0

    folders = [int(name) for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
    
    if folders:
        return max(folders) + 1
    else:
        return 0
    
def file_work(LOG_DIR):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    files_to_copy = [
        '/home/ubuntu/lcl/cvpr2025/classification/configs/config_DVSCifar10.yaml',
        '/home/ubuntu/lcl/cvpr2025/classification/data/loader.py',
        '/home/ubuntu/lcl/cvpr2025/classification/train_DVSCifar10.py',
        '/home/ubuntu/lcl/cvpr2025/classification/data/cifar10_dvs.py',
        '/home/ubuntu/lcl/cvpr2025/classification/model_ptv/get_model.py',
        '/home/ubuntu/lcl/cvpr2025/classification/model_ptv/dela.py',
        '/home/ubuntu/lcl/cvpr2025/classification/model_ptv/EP2T.py',
        '/home/ubuntu/lcl/cvpr2025/classification/model_ptv/event_utils.py',
        '/home/ubuntu/lcl/cvpr2025/classification/PointTransformerV3/model_ptv3.py'
    ]

    for file_path in files_to_copy:
        if glob.glob(file_path):
            shutil.copy(glob.glob(file_path)[0], LOG_DIR)

def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--validation_dataset", default='/home/lxh/LCL/cvpr2025/classification/data/dataset/cifar10_dvs')
    parser.add_argument("--training_dataset", default='/home/lxh/LCL/cvpr2025/classification/data/dataset/cifar10_dvs')
    parser.add_argument('--config_path', default='/home/ubuntu/lcl/cvpr2025/classification/output/DVSCifar10/log/49/config_DVSCifar10.yaml')


    # logging options
    

    cur_id = find_next_folder_index('/home/ubuntu/lcl/cvpr2025/classification/output/DVSCifar10/log')
    # cur_id = 5001
    print(cur_id)

    parser.add_argument("--log_dir", default=os.path.join('/home/ubuntu/lcl/cvpr2025/classification/output/DVSCifar10/log', 
                                         str(cur_id)))
    parser.add_argument("--log_tensorboard", default=os.path.join('/home/ubuntu/lcl/cvpr2025/classification/output/DVSCifar10/tensorboard', 
                                         str(cur_id)))
    parser.add_argument("--model_save", default=os.path.join('/home/ubuntu/lcl/cvpr2025/classification/output/DVSCifar10/model', 
                                         str(cur_id)))
    parser.add_argument("--checkpoint", default='/home/ubuntu/lcl/cvpr2025/classification/output/DVSCifar10/model/23/model_best.pth')
    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=10)

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)

    flags = parser.parse_args()
    config = load_yaml(flags.config_path)
    config.update(vars(flags))

    if os.path.exists(flags.log_dir) == False:
            os.makedirs(flags.log_dir, exist_ok=True)
    if os.path.exists(flags.model_save) == False:
            os.makedirs(flags.model_save, exist_ok=True)
    file_work(os.path.join('/home/ubuntu/lcl/cvpr2025/classification/output/DVSCifar10/log',str(cur_id)))

    print(f"----------------------------\n"
          f"Starting training with \n"
          f"num_epochs: {flags.num_epochs}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"log_dir: {flags.log_dir}\n"
          f"training_dataset: {flags.training_dataset}\n"
          f"validation_dataset: {flags.validation_dataset}\n"
          f"----------------------------")
    
    return EasyDict(config)

if __name__ == '__main__':
    set_seed(4)
    torch.set_printoptions(threshold=float('inf'), precision=4, sci_mode=False)
    configs = FLAGS()    
    logfile = "{dir}/out2.log".format(dir = configs.log_dir)
    errfile = "{dir}/err2.log".format(dir = configs.log_dir)
    logfile = open(logfile, "a", 1)
    errfile = open(errfile, "a", 1)
    # sys.stdout = logfile
    # sys.stderr = errfile

    # datasets, add augmentation to training set
    training_dataset = DVSCifar10(configs, mode='training')
    validation_dataset = DVSCifar10(configs, mode='testing')

    training_loader = Loader(training_dataset, 
            batch_size = configs['dataset_params']['train_data_loader']['batch_size'], 
            num_workers = configs['dataset_params']['train_data_loader']['num_workers'], 
            pin_memory = True, 
            device = configs['dataset_params']['train_data_loader']['device'])

    validation_loader = Loader(validation_dataset, 
            batch_size = configs['dataset_params']['val_data_loader']['batch_size'], 
            num_workers = configs['dataset_params']['val_data_loader']['num_workers'], 
            pin_memory = True, 
            device = configs['dataset_params']['val_data_loader']['device'])


    model = get_model(configs['dataset_params']['num_classes'])

    # ckpt = torch.load('/home/ubuntu/lcl/cvpr2025/classification/output/DVSCifar10/model/23/model_best.pth')
    # state_dict = ckpt["state_dict"]
    # new_state_dict = collections.OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # 去掉前缀（去掉前七个字符）
    #     new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    # model.load_state_dict(new_state_dict, strict=True)  # 重新加载这个模型

    model = DataParallel(model).cuda()
    criterion = cross_entropy_loss_and_accuracy 

    # optimizer and lr scheduler
    if configs['train_params']['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(configs['train_params']["learning_rate"]), betas=configs['train_params']['betas'])
    elif configs['train_params']['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(configs['train_params']["learning_rate"]), momentum=configs['train_params']['momentum'])
    elif configs['train_params']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(configs['train_params']["learning_rate"]), weight_decay=configs['train_params']['weight_decay'])
    if configs['train_params']['lr_scheduler'] == 'ExponentialLR':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, configs['train_params']['lr_lambda'])
    elif configs['train_params']['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
            T_0=configs['train_params']['T0'], T_mult = configs['train_params']['T_mult'], eta_min=configs['train_params']['final_lr'])
    elif configs['train_params']['lr_scheduler'] == 'OnecycleLR':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.002, pct_start = 0.04, anneal_strategy = 'cos',div_factor = 10.0, epochs=120, steps_per_epoch=900,
                final_div_factor = 100.0,)
    writer = SummaryWriter(configs['log_tensorboard'])

    iteration = 0
    min_validation_loss = 1000
    min_train_loss = 1000
    max_validation_accuracy = 0
    lr_cycle = 0
    

    for i in range(configs['train_params']['max_num_epochs']):
        if  i != 0 and i % configs['train_params']['save_every_n_epochs'] == 0:
            sum_accuracy = 0
            sum_loss = 0
            model = model.eval()
            correct = torch.zeros(configs['dataset_params']['num_classes']).cuda()
            total = torch.zeros(configs['dataset_params']['num_classes']).cuda()
            print(f"Validation step [{i:3d}/{configs['train_params']['max_num_epochs']:3d}]")
            for fus, events, events_ptv3, coord, labels, grid_size, batch in tqdm.tqdm(validation_loader):
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
                    pred_labels = model(dict_event)
                    loss, accuracy = criterion(pred_labels, labels.long())
                sum_accuracy += accuracy
                sum_loss += loss

                _, predicted = torch.max(pred_labels, 1)

                for j in range(configs['dataset_params']['num_classes']):
                    correct[j] += (predicted[labels == j] == j).sum().item()
                    total[j] += (labels == j).sum().item()
                
            validation_loss = sum_loss.item() / len(validation_loader)
            validation_accuracy = sum_accuracy.item() / len(validation_loader)

            class_accuracies = correct / total
            class_accuracies[total == 0] = 0
            accuracy_output = ""
            for j in range(configs['dataset_params']['num_classes']):
                accuracy_output += f'Class {j}: {class_accuracies[j].item() * 100:.0f}% --- '
            print(accuracy_output[:-5])

            writer.add_scalar("validation/accuracy", validation_accuracy, iteration)
            writer.add_scalar("validation/loss", validation_loss, iteration)

            print(f"Validation Loss {validation_loss:.4f}  Accuracy {validation_accuracy:.4f}")

            # lr_cycle += 1

            if validation_accuracy > max_validation_accuracy:
                max_validation_accuracy = validation_accuracy
                state_dict = model.state_dict()
                # lr_scheduler.step()
                # lr_cycle = 0

                torch.save({
                    "state_dict": state_dict,
                    "min_val_loss": min_validation_loss,
                    "iteration": iteration
                }, os.path.join(configs['model_save'],"model_best.pth"))
                print("New best at accuracy", validation_accuracy)

            # elif lr_cycle % 9 == 8:
            #     lr_scheduler.step()
            #     lr_cycle = 0


        sum_accuracy = 0
        sum_loss = 0
        model = model.train()
        correct = torch.zeros(configs['dataset_params']['num_classes']).cuda()
        total = torch.zeros(configs['dataset_params']['num_classes']).cuda()
        print(f"Training step [{i:3d}/{configs['train_params']['max_num_epochs']:3d}]")
        import time
        last_loader_time = time.time()
        for fus, events, events_ptv3, coord, labels, grid_size, batch in tqdm.tqdm(training_loader): #events.shape:(B,N,C)
            # print(f"Loader time: {time.time() - last_loader_time:.4f}")
            
            optimizer.zero_grad()
            fus = fus.cuda()
            events = events.cuda()
            coord = coord.cuda()
            events_ptv3 = events_ptv3.cuda()
            labels = labels.cuda()
            batch = batch.cuda()
            grid_size = grid_size.cuda()
            dict_event = {
                'coord': coord,
                'feat': events_ptv3,
                'batch': batch,
                'fus': fus,
                'feat_ori': events,
                'grid_size': grid_size
                }
            last_model_time = time.time()
            pred_labels = model(dict_event)

            _, predicted = torch.max(pred_labels, 1)
            for j in range(configs['dataset_params']['num_classes']):
                correct[j] += (predicted[labels == j] == j).sum().item()
                total[j] += (labels == j).sum().item()

            loss, accuracy = criterion(pred_labels, labels.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            sum_accuracy += accuracy
            sum_loss += loss
    
            iteration += 1
            last_loader_time = time.time()
            


        if i % 7 == 6:
            lr_scheduler.step()
        

        class_accuracies = correct / total
        class_accuracies[total == 0] = 0
        accuracy_output = ""
        for j in range(configs['dataset_params']['num_classes']):
            accuracy_output += f'Class {j}: {class_accuracies[j].item() * 100:.0f}% --- '
        print(accuracy_output[:-5])

        training_loss = sum_loss.item() / len(training_loader)
        training_accuracy = sum_accuracy.item() / len(training_loader)
        print(f"Training Iteration {iteration:5d}  Loss {training_loss:.4f}  Accuracy {training_accuracy:.4f}")

        writer.add_scalar("training/accuracy", training_accuracy, iteration)
        writer.add_scalar("training/loss", training_loss, iteration)
        if training_loss < min_train_loss:
                min_train_loss = training_loss
                state_dict = model.state_dict()

                torch.save({
                    "state_dict": state_dict,
                    "min_val_loss": min_train_loss,
                    "iteration": iteration
                }, os.path.join(configs['model_save'],"trainloss_best.pth"))
                # print("New best at loss", training_loss)