import torch
import os, shutil
import numpy as np
from tqdm import tqdm
from models.unet import ResUNet, UNet, CBAMUNet, ResUNet1
from torch.nn import functional as F
from torch.utils.data import DataLoader
from PIL import Image
from data.TongueDataset import TongueDataset1, RandomTransform
from util import Metrics, save_seg_images, save_model, load_model_weights, calculate_iou, calculate_dice
from torch.utils.tensorboard import SummaryWriter

np.set_printoptions(precision=5)

def process(model, dataloader, device):
    model.eval()  # 设置模型为评估模式
    
    progress_bar = tqdm(dataloader, desc=f'   test', leave=False)
    with torch.no_grad():  # 关闭梯度计算
        for image, mask_path in progress_bar:
            image = image.to(device)
            image = downsample(image)
            
            # 前向传播
            outputs = model(image)
            outputs = upsample(outputs)
            pred_masks = (outputs > 0.8).float()
            
            pred_masks = pred_masks.squeeze(1).cpu().numpy()[0]
            
            mask_image = Image.fromarray(pred_masks).convert("L")  # "L"模式表示单通道灰度图
    
            # 保存为PNG
            mask_image.save(mask_path[0], format='PNG')
            
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            progress_bar.set_postfix({
                'GPU': f"{memory_allocated:.0f} MB",
                })
            # break
            
def upsample(tensor, size=(800, 800)):
    return F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)

def downsample(tensor, size=(320, 320)):
    return F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)

if __name__ == '__main__':
    # 创建数据集实例
    
    # dataset = TongueDataset('data/Tongue_test', transform=RandomTransform())
    train_set = TongueDataset1('data/Tongue_train')#, transform=RandomTransform())
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    test_set = TongueDataset1('data/Tongue_test')#, transform=RandomTransform())
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = ResUNet1(layers=[2,2,2]).to(device)
    load_model_weights(model, 'runs/res222_aspp04/unet_epoch100-109.pt')

    # load_model_weights(model, 'weights/unet_exp1.pt')
    process(model, train_loader, device = device)
    process(model, test_loader, device = device)
