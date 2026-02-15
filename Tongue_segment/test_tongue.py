import torch
import os, shutil
import numpy as np
from tqdm import tqdm
from models.unet import ResUNet, UNet, CBAMUNet, ResUNet2
from torch.nn import functional as F
from torch.utils.data import DataLoader
from PIL import Image
from data.TongueDataset import TongueDataset, RandomTransform
from util import Metrics, save_seg_images, save_model, load_model_weights, calculate_iou, calculate_dice
from torch.utils.tensorboard import SummaryWriter

np.set_printoptions(precision=5)

def test_tongue(model, dataloader, device):
    model.eval()  # 设置模型为评估模式
    results = []
    main_t = 0
    second_t = 0
    top3_t = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f'   test', leave=False)
    with torch.no_grad():  # 关闭梯度计算
        for images, masks, features in progress_bar:
            images, masks = images.to(device), masks.to(device)
            features = features[:, :33]
            feature_topk = np.flip(np.argsort(features[0]).numpy())
            # 前向传播
            outputs, pred = model(images)
            pred = pred.cpu().numpy()
            pred_topk = np.flip(np.argsort(pred[0]))
            # pred_masks = (outputs > 0.8).float()
            # pred_masks = pred_masks.squeeze(1).cpu().numpy()[0]
            # mask_image = Image.fromarray(pred_masks).convert("L")  # "L"模式表示单通道灰度图
            pred = np.round(pred * 4)
            
            print(pred, features)
            if pred_topk[0] == feature_topk[0]:
                main_t += 1
            if pred_topk[1] == feature_topk[1] or pred_topk[0] == feature_topk[1]:
                second_t += 1
            if pred_topk[0] in feature_topk[:3]:
                top3_t += 1
                
            total += 1
            results.append(pred)
            
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            progress_bar.set_postfix({
                'GPU': f"{memory_allocated:.0f} MB",
                })
            # break
        print(main_t/total, second_t/total, top3_t/total)
            
def upsample(tensor, size=(800, 800)):
    return F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)

def downsample(tensor, size=(320, 320)):
    return F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)

if __name__ == '__main__':
    # 创建数据集实例
 
    test_set = TongueDataset('data/Tongue_test', transform=RandomTransform(brightness_range=(0.99, 1.01), contrast_range=(0.99, 1.01), hue_range=(-0.01, 0.01)))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = ResUNet2(num_classes=33).to(device)
    load_model_weights(model, 'weights/tongue05.pt')

    # load_model_weights(model, 'weights/unet_exp1.pt')
    test_tongue(model, test_loader, device = device)
