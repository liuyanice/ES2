import os, argparse, math
import numpy as np
from glob import glob
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import time

from medpy.metric.binary import hd, dc, assd, jc
from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance
from torch.utils.tensorboard import SummaryWriter
from lib.sampling_points import sampling_points, point_sample
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR



def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='bgctrans')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--net_layer', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='isic2016')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--lr_seg', type=float, default=1e-4)  #0.0003
    parser.add_argument('--n_epochs', type=int, default=150)  #100
    parser.add_argument('--bt_size', type=int, default=8)  #36
    parser.add_argument('--seg_loss', type=int, default=1, choices=[0, 1])
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--patience', type=int, default=500)  #50

    #cross-transformer
    parser.add_argument('--ne_num', type=int, default=2)
    parser.add_argument('--md_num', type=int, default=2)
    parser.add_argument('--fusion', type=int, default=1)
    parser.add_argument('--w', type=float, default=1)

    #log_dir name
    parser.add_argument('--folder_name', type=str, default='0')#Default_folder

    parse_config = parser.parse_args()
    print(parse_config)
    return parse_config


def ce_loss(pred, gt):
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    return (-gt * torch.log(pred) - (1 - gt) * torch.log(1 - pred)).mean()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
    
def dice_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()

#-------------------------- train func --------------------------#
def train(epoch):
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        m_i, m, rend, points= model(data)
        gt_points = point_sample(
                label.to(torch.float),
                points,
                mode="nearest",
                align_corners=False
                )
            
        gt_points=gt_points.squeeze(1).to(torch.long)
        if parse_config.ne_num + parse_config.md_num > 0:
            seg_loss = 0.0
            for i in m_i:
                seg_loss = seg_loss + structure_loss(i, label)
            seg_loss = seg_loss / len(m_i)+structure_loss(m, label)
            loss = seg_loss +parse_config.w*F.cross_entropy(rend, gt_points, ignore_index=255)
            if batch_idx % 50 == 0:
                show_image = [label[0], F.sigmoid(m_i[0][0])]
                show_image = torch.cat(show_image, dim=2)
                show_image = show_image.repeat(3, 1, 1)
                show_image = torch.cat([data[0], show_image], dim=2)

                writer.add_image('pred/all',
                                     show_image,
                                     epoch * len(train_loader) + batch_idx,
                                     dataformats='CHW')
        else:
            seg_loss = 0.0
            for i in m_i:
                seg_loss = seg_loss + structure_loss(i, label)
            seg_loss = seg_loss / len(m_i)+structure_loss(m, label)
            loss = seg_loss +parse_config.w*F.cross_entropy(rend, gt_points, ignore_index=255)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t[lateral-2: {:.4f}, lateral-3: {:0.4f}]'#, lateral-4: {:0.4f}
                .format(epoch, batch_idx * len(data),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss,
                        seg_loss))


#-------------------------- eval func --------------------------#
def evaluation(epoch, loader):
    model.eval()
    dice_value = 0
    iou_value = 0
    dice_average = 0
    iou_average = 0
    numm = 0
    for batch_idx, batch_data in enumerate(loader):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        with torch.no_grad():
            if parse_config.arch == 'transfuse':
                _, _, m = model(data)
                loss_fuse = structure_loss(m, label)
            elif parse_config.arch == 'bgctrans':
                m_i, m, rend, points = model(data)
                loss = 0
            if parse_config.arch == 'transfuse':
                loss = loss_fuse
            m = m.cpu().numpy() > 0.5
        label = label.cpu().numpy()
        assert (m.shape == label.shape)
        dice_ave  = dc(m, label)
        iou_ave =  jc(m, label)
        dice_value += dice_ave
        iou_value += iou_ave
        numm += 1

    dice_average = dice_value / numm
    iou_average = iou_value / numm
    writer.add_scalar('val_metrics/val_dice', dice_average, epoch)
    writer.add_scalar('val_metrics/val_iou', iou_average, epoch)
    print("Average dice value of evaluation dataset = ", dice_average)
    print("Average iou value of evaluation dataset = ", iou_average)
    return dice_average, iou_average, loss


if __name__ == '__main__':
    #-------------------------- get args --------------------------#
    parse_config = get_cfg()

    #-------------------------- build loggers and savers --------------------------#
    exp_name = parse_config.dataset + '/' + parse_config.exp_name + '_loss_' + str(
        parse_config.seg_loss) + '_aug_' + str(
            parse_config.aug
        ) + '/' + parse_config.folder_name + '/fold_' + str(parse_config.fold)

    os.makedirs('logs/{}'.format(exp_name), exist_ok=True)
    os.makedirs('logs/{}/model'.format(exp_name), exist_ok=True)
    writer = SummaryWriter('logs/{}/log'.format(exp_name))
    save_path = 'logs/{}/model/best.pkl'.format(exp_name)
    latest_path = 'logs/{}/model/latest.pkl'.format(exp_name)

    EPOCHS = parse_config.n_epochs
    os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu
    device_ids = range(torch.cuda.device_count())

    #-------------------------- build dataloaders --------------------------#
    if parse_config.dataset == 'isic2018':
        from utils.isic2018_dataset import norm01, myDataset

        dataset = myDataset(fold=parse_config.fold,
                            split='train',
                            aug=parse_config.aug)
        dataset2 = myDataset(fold=parse_config.fold, split='valid', aug=False)
    elif parse_config.dataset == 'isic2016':
        from utils.isic2016_dataset import norm01, myDataset

        dataset = myDataset(split='train', aug=parse_config.aug)
        dataset2 = myDataset(split='valid', aug=False)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=parse_config.bt_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset2,
        batch_size=1,  #parse_config.bt_size
        shuffle=False,  #True
        num_workers=2,
        pin_memory=True,
        drop_last=False)  #True

    #-------------------------- build models --------------------------#
    if parse_config.arch is 'bgctrans':
        from lib.bgctrans import load_pvtv2
        model = load_pvtv2(1, parse_config.ne_num, parse_config.md_num,
                            parse_config.fusion, 352).cuda()#352
    elif parse_config.arch == 'transfuse':
        from lib.TransFuse.TransFuse import TransFuse_S
        model = TransFuse_S(pretrained=True).cuda()

    if len(device_ids) > 1:  # 多卡训练
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr_seg)

    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)#T_max=18

    criteon = [None, ce_loss][parse_config.seg_loss]

    #--------------------------Training --------------------------#

    max_dice = 0
    max_iou = 0
    best_ep = 0

    min_loss = 10
    min_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        start = time.time()
        train(epoch)
        dice, iou, loss = evaluation(epoch, val_loader)
        scheduler.step()

        if loss < min_loss:
            min_epoch = epoch
            min_loss = loss
        else:
            if epoch - min_epoch >= parse_config.patience:
                print('Early stopping!')
                break
        if iou > max_iou:
            max_iou = iou
            best_ep = epoch
            torch.save(model.state_dict(), save_path)
        else:
            if epoch - best_ep >= parse_config.patience:
                print('Early stopping!')
                break
        torch.save(model.state_dict(), latest_path)
        time_elapsed = time.time() - start
        print(
            'Training and evaluating on epoch:{} complete in {:.0f}m {:.0f}s'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))
