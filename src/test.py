import os, argparse, sys, tqdm, logging, cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from glob import glob
from medpy.metric.binary import hd, hd95, dc, jc, assd
from lib.bgctrans import load_pvtv2
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import convolve


parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='bgctrans')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--net_layer', type=int, default=50)
parser.add_argument('--dataset', type=str, default='isic2016')
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--fold', type=str, default='0')
parser.add_argument('--lr_seg', type=float, default=1e-4)  
parser.add_argument('--n_epochs', type=int, default=200)  
parser.add_argument('--bt_size', type=int, default=1)  
parser.add_argument('--seg_loss', type=int, default=1, choices=[0, 1])
parser.add_argument('--aug', type=int, default=1)
parser.add_argument('--patience', type=int, default=500)  

# cross-transformer
parser.add_argument('--ne_num', type=int, default=2)
parser.add_argument('--md_num', type=int, default=2)
parser.add_argument('--fusion', type=int, default=1)


#log_dir name
parser.add_argument('--folder_name', type=str, default='0')#Default_folder

parse_config = parser.parse_args()
print(parse_config)
os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if parse_config.dataset == 'isic2018':
    from utils.isic2018_dataset import norm01, myDataset
    dataset = myDataset(parse_config.fold, 'valid', aug=False)
elif parse_config.dataset == 'isic2016':
    from utils.isic2016_dataset import norm01, myDataset
    dataset = myDataset('valid', aug=False)

if parse_config.arch is 'bgctrans':
    from lib.bgctrans import load_pvtv2
    model = load_pvtv2(1, 2, 2, 1, 352).to(device)
else:
    #from Ours.base import DeepLabV3
    #model = DeepLabV3(1, parse_config.net_layer).cuda()
    from lib.TransFuse.TransFuse import TransFuse_S
    model = TransFuse_S(pretrained=True).cuda()
    
def fspecial_gauss(size, sigma):
       """Function to mimic the 'fspecial' gaussian MATLAB function
       """
       x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
       g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
       return g/g.sum()
       
def original_WFb(pred, gt):
    E = np.abs(pred - gt)
    dst, idst = distance_transform_edt(1 - gt, return_indices=True)

    K = fspecial_gauss(7, 5)
    Et = E.copy()
    Et[gt != 1] = Et[idst[:, gt != 1][0], idst[:, gt != 1][1]]
    EA = convolve(Et, K, mode='nearest')
    MIN_E_EA = E.copy()
    MIN_E_EA[(gt == 1) & (EA < E)] = EA[(gt == 1) & (EA < E)]

    B = np.ones_like(gt)
    B[gt != 1] = 2.0 - 1 * np.exp(np.log(1 - 0.5) / 5 * dst[gt != 1])
    Ew = MIN_E_EA * B

    TPw = np.sum(gt) - np.sum(Ew[gt == 1])
    FPw = np.sum(Ew[gt != 1])

    R = 1 - np.mean(Ew[gt == 1])
    P = TPw / (TPw + FPw + np.finfo(np.float64).eps)
    Q = 2 * R * P / (R + P + np.finfo(np.float64).eps)

    return Q


test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)


def test():
    model = _segm_pvtv2(1, 2, 2, 1, 352).to(device)
    model.eval()
    num = 0
    dice_value = 0
    jc_value = 0
    hd95_value = 0
    assd_value = 0
    f_value = 0
    from tqdm import tqdm
    labels = []
    pres = []
    for batch_idx, batch_data in tqdm(enumerate(test_loader)):
        model.load_state_dict(
            torch.load(
                r'root\BGC-Trans\src\logs\isic2016\test_loss_1_aug_1\0\fold_0\model\latest.pkl'
            ))
        
        data = batch_data['image'].to(device).float()
       # print("sssss",data.shape)
        label = batch_data['label'].to(device).float()
        with torch.no_grad():
       
            output, pixoutput, rend, points = model(data)
            m = pixoutput.cpu().numpy() > 0.5
            
        label = label.cpu().numpy()
        assert (m.shape == label.shape)
        labels.append(label)
        pres.append(m)
    labels = np.concatenate(labels, axis=0)
    pres = np.concatenate(pres, axis=0)
    print(labels.shape, pres.shape)
    for _id in range(labels.shape[0]):
        dice_ave = dc(labels[_id], pres[_id])
        jc_ave = jc(labels[_id], pres[_id])
        f_ave = original_WFb(pres[_id].squeeze(0), labels[_id].squeeze(0))
        try:
            hd95_ave = hd95(labels[_id], pres[_id])
            assd_ave = assd(labels[_id], pres[_id])
        except RuntimeError:
            num += 1
            hd95_ave = 0
            assd_ave = 0
            f_ave = 0
        dice_value += dice_ave
        jc_value += jc_ave
        hd95_value += hd95_ave
        assd_value += assd_ave
        f_value += f_ave
    dice_average = dice_value / (labels.shape[0] - num)
    jc_average = jc_value / (labels.shape[0] - num)
    hd95_average = hd95_value / (labels.shape[0] - num)
    assd_average = assd_value / (labels.shape[0] - num)
    f_average = f_value / (labels.shape[0] - num)
    logging.info('Dice value of test dataset  : %f' % (dice_average))
    logging.info('Jc value of test dataset  : %f' % (jc_average))
    logging.info('Hd95 value of test dataset  : %f' % (hd95_average))
    logging.info('Assd value of test dataset  : %f' % (assd_average))
    logging.info('f value of test dataset  : %f' % (f_average))
    print("Average dice value of evaluation dataset = ", dice_average)
    print("Average jc value of evaluation dataset = ", jc_average)
    print("Average hd95 value of evaluation dataset = ", hd95_average)
    print("Average assd value of evaluation dataset = ", assd_average)
    print("Average f value of evaluation dataset = ", f_average)
    return dice_average


if __name__ == '__main__':
    test()