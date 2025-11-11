import sys
sys.path.append("/home/ubuntu/lcl/cvpr2025/classification")
from model_ptv.EP2T import EP2T
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet101 ,resnet34, resnet18
from torchvision.models.vision_transformer import vit_b_16
import timm

class Resnet(nn.Module):
    def __init__(self, c=64, n = 101) -> None:
        super().__init__()
        self.resnet34 = resnet34(pretrained=True)
        self.resnet34.conv1 = nn.Conv2d(c, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet34.fc = nn.Linear(512, n)
    def forward(self, input):
        x = self.resnet34(input)
        return x

class vision_transformer(nn.Module):
    def __init__(self, c = 64, n = 101)-> None:
        super().__init__()
        self.vision_transformer = vit_b_16(pretrained=True)
        self.vision_transformer.conv_proj = nn.Conv2d(c, 768, kernel_size=(16, 16), stride=(16, 16))
        self.vision_transformer.head = nn.Linear(768, n)
    def forward(self, input):
        x = self.vision_transformer(input)
        return x


class get_model(nn.Module):
    def __init__(self, class_num) -> None:
        super().__init__()
        
        self.ep2t = EP2T(pixel_size = (128, 128), embedding_size=64)
        self.fc = Resnet(c = 29 + 64,n = class_num)
        
    def forward(self, dict_event):
        fus = dict_event['fus']
        x = self.ep2t(dict_event)
        x = torch.cat([fus, x], dim=1)
        
        x = self.fc(x)
        
        return x
    
class get_model_caltech101(nn.Module):
    def __init__(self, class_num) -> None:
        super().__init__()
        self.fc = Resnet(c = 64 + 29,n = class_num)
        # self.fc = efficientnet(c = 64 + 29, n = class_num)
        self.ep2t = EP2T(pixel_size = (240, 180), embedding_size=64)
        # self.att = SELay(c=[64 + 29, 16])
        # self.att2 = SELay2(c=[64, 16])
        
    def forward(self, dict_event):
        fus = dict_event['fus']
        x = self.ep2t(dict_event)
        x = torch.cat([fus, x], dim=1)
        x = self.fc(x)
        return x
    
    
if __name__ == "__main__":

    model = get_model(class_num=10)  # 替换为你的模型
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params}")