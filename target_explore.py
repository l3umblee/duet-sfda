import os
import os.path as osp
import numpy as np
import torch
import random
import pickle
import cv2
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from src.utils import loss
from src.models import network
from src.data.data_list import ImageList, ImageList_idx
from conf import cfg, load_cfg_from_args
from torchvision.transforms import ToPILImage
from metric_func import *
from visualize_func import *


# logger = logging.getLogger(__name__)

def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    #   else:
    #     normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    #   else:
    #     normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(cfg):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_tar = open(cfg.t_dset_path).readlines()  # target dataset
    txt_test = open(cfg.t_dset_path).readlines()  # target dataset

    # if not cfg.da == 'uda':
    #     label_map_s = {}
    #     for i in range(len(cfg.src_classes)):
    #         label_map_s[cfg.src_classes[i]] = i

    #     new_tar = []
    #     for i in range(len(txt_tar)):
    #         rec = txt_tar[i]
    #         reci = rec.strip().split(' ')
    #         if int(reci[1]) in cfg.tar_classes:
    #             if int(reci[1]) in cfg.src_classes:
    #                 line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
    #                 new_tar.append(line)
    #             else:
    #                 line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
    #                 new_tar.append(line)
    #     txt_tar = new_tar.copy()
    #     txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=cfg.NUM_WORKERS,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    # target_classes = [1, 3, 20]
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            # labels = labels.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)

            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()

                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output_pred = nn.Softmax(dim=1)(all_output).cpu()
    # classic_ece(all_output_pred, all_label, cfg.MODEL.METHOD)
    # visualize_latent_space(all_fea, all_label, cfg.MODEL.METHOD, 0, )
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


if __name__ == "__main__":
    load_cfg_from_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    print(f'cfg.GPU_ID: {cfg.GPU_ID}')

    cfg.type = cfg.domain
    cfg.t_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.T] + '_list.txt'
    cfg.test_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.T] + '_list.txt'
    cfg.s_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.S] + '_list.txt'

    if cfg.SETTING.DATASET == 'office-home':
        if cfg.DA == 'pda':
            cfg.class_num = 65
            cfg.src_classes = [i for i in range(65)]
            cfg.tar_classes = [i for i in range(25)]

    dset_loaders = data_load(cfg)
    print(f"cfg : {cfg}")

    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num=cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()

    netF.eval()
    netB.eval()
    netC.eval()

    model_path = osp.join(cfg.output_dir + '/target_F_' + cfg.MODEL.METHOD + '.pt')
    netF.load_state_dict(torch.load(model_path))
    model_path = osp.join(cfg.output_dir + '/target_B_' + cfg.MODEL.METHOD + '.pt')
    netB.load_state_dict(torch.load(model_path))
    model_path = osp.join(cfg.output_dir + '/target_C_' + cfg.MODEL.METHOD + '.pt')
    netC.load_state_dict(torch.load(model_path))

    for i in range(len(cfg.domain)):
        if i != cfg.SETTING.S:
            continue
        cfg.SETTING.T = i
        dset_loaders = data_load(cfg)
        if cfg.SETTING.DATASET == 'VISDA-C':
            acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(cfg.SOURCE.TRTE, cfg.name,
                                                                            acc) + '\n' + acc_list

        else:
            acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(cfg.SOURCE.TRTE, cfg.name, acc)

        # logging.info(log_str)
        print(f"log_str: {log_str}")