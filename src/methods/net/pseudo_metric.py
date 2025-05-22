import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.models import network, shot_model
from sklearn.metrics import confusion_matrix
from src.utils import loss
from src.data.datasets.data_loading import get_test_loader
from src.data.datasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK, IMAGENET_V_MASK
from src.models.model import *


class MetricProjection(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        self.D = latent_dim

        metric_tensor = torch.randn(latent_dim, latent_dim)
        squared_sum = torch.sum(metric_tensor ** 2)
        normalized_metric_tensor = metric_tensor / torch.sqrt(squared_sum)

        self.metric_tensor = nn.Parameter(normalized_metric_tensor)

    def forward(self, feas, prototypes):
        bs = feas.size(0)
        v = feas - prototypes

        metric_tensor =self.metric_tensor * self.metric_tensor.T
        metric_tensor = nn.Softmax(dim=1)(metric_tensor)

        dist = torch.einsum('bi,ii,ib->b', v, metric_tensor, v.T)
        # dist = torch.matmul(v, v.T)
        mean_dist = dist.sum() / bs

        # sing_val = torch.svd(metric_tensor / np.sqrt(metric_tensor.shape[0]))[1]
        # eig_val = sing_val ** 2
        # von_loss = - (eig_val * torch.log(eig_val)).nansum()
        # loss = mean_dist - von_loss
        return mean_dist


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    # decay = 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def cal_acc(loader, base_model, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = base_model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
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


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def cosine_scheduler(cfg, optimizer, iter_num, max_iter, lr_min=1e-6):
    for param_group in optimizer.param_groups:
        lr_max = param_group['lr0']  # Initial learning rate
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * iter_num / max_iter))
        param_group['lr'] = lr
        param_group['weight_decay'] = cfg.OPTIM.WD
        param_group['momentum'] = cfg.OPTIM.MOMENTUM
        param_group['nesterov'] = cfg.OPTIM.NESTEROV
    return optimizer


def train_target(cfg):
    ## set base network
    if 'image' in cfg.SETTING.DATASET:
        if cfg.MODEL.ARCH[0:3] == 'res':
            netF = network.ResBase(res_name=cfg.MODEL.ARCH)
        elif cfg.MODEL.ARCH[0:3] == 'vgg':
            netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH)
        netC = network.Net2(2048, 1000)
        base_model = get_model(cfg, cfg.class_num)
        netC.linear.load_state_dict(base_model.model.fc.state_dict())
        del base_model
        Shot_model = shot_model.OfficeHome_Shot(netF, netC)
        base_model = Shot_model
        if cfg.SETTING.DATASET == "imagenet_a":
            base_model = ImageNetXWrapper(base_model, IMAGENET_A_MASK)
        elif cfg.SETTING.DATASET == "imagenet_r":
            base_model = ImageNetXWrapper(base_model, IMAGENET_R_MASK)
        elif cfg.SETTING.DATASET == "imagenet_d109":
            base_model = ImageNetXWrapper(base_model, IMAGENET_D109_MASK)
        elif cfg.SETTING.DATASET == "imagenet_v":
            base_model = ImageNetXWrapper(base_model, IMAGENET_V_MASK)
    else:
        base_model = get_model(cfg, cfg.class_num)
    base_model = base_model.cuda()

    latent_dim = base_model.fc.weight.size(-1)
    proj_head = MetricProjection(latent_dim).cuda()

    param_group = []
    for k, v in base_model.named_parameters():
        if 'netC' in k or 'fc' in k:
            v.requires_grad = False
        else:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR}]

    for k, v in proj_head.named_parameters():
        param_group += [{'params': v, 'lr': cfg.OPTIM.LR}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    cfg.ADAPTATION = 'plmatch'
    domain_name = cfg.domain[cfg.SETTING.T]
    target_data_loader = get_test_loader(adaptation=cfg.ADAPTATION,
                                         dataset_name=cfg.SETTING.DATASET,
                                         root_dir=cfg.DATA_DIR,
                                         domain_name=domain_name,
                                         rng_seed=cfg.SETTING.SEED,
                                         batch_size=cfg.TEST.BATCH_SIZE,
                                         shuffle=True,
                                         workers=cfg.NUM_WORKERS)

    test_data_loader = get_test_loader(adaptation=cfg.ADAPTATION,
                                       dataset_name=cfg.SETTING.DATASET,
                                       root_dir=cfg.DATA_DIR,
                                       domain_name=domain_name,
                                       rng_seed=cfg.SETTING.SEED,
                                       batch_size=cfg.TEST.BATCH_SIZE * 3,
                                       shuffle=False,
                                       workers=cfg.NUM_WORKERS)

    base_model.eval()

    max_iter = cfg.TEST.MAX_EPOCH * len(target_data_loader)
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(target_data_loader)
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test[0].size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and cfg.SHOT.CLS_PAR > 0:
            base_model.eval()
            mixed_soft = obtain_label(test_data_loader, base_model, proj_head, cfg)
            mixed_soft = mixed_soft.cuda()
            base_model.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        weak_x = inputs_test[1].cuda()
        strong_x = inputs_test[2].cuda()

        if 'image' in cfg.SETTING.DATASET:
            weak_feas = base_model.netF(weak_x)
            strong_feas = base_model.netF(strong_x)
        else:
            weak_feas = base_model.encoder(weak_x)
            strong_feas = base_model.encoder(strong_x)

        metric_loss = proj_head(weak_feas, strong_feas)

        if 'image' in cfg.SETTING.DATASET:
            if 'k' in cfg.SETTING.DATASET:
                weak_outputs = base_model.netC(weak_feas)
            else:
                weak_outputs = base_model.masking_layer(base_model.netC(weak_feas))
        else:
            weak_outputs = base_model.fc(weak_feas)

        if cfg.SHOT.CLS_PAR > 0:
            outputs_pred = nn.Softmax(dim=1)(weak_outputs)
            batch_soft = mixed_soft[tar_idx]

            classifier_loss = F.kl_div(outputs_pred.log(), batch_soft, reduction='batchmean')
            classifier_loss *= cfg.SHOT.CLS_PAR
            if iter_num < interval_iter and cfg.SETTING.DATASET == 'VISDA-C':
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if cfg.SHOT.ENT:
            softmax_out = nn.Softmax(dim=1)(weak_outputs)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if cfg.SHOT.GENT:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.SHOT.EPSILON))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * cfg.SHOT.ENT_PAR
            classifier_loss += im_loss
        classifier_loss += metric_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_model.eval()
            acc_s_te, _ = cal_acc(test_data_loader,base_model,False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te)
            logging.info(log_str)
            base_model.train()

    if cfg.ISSAVE:
        torch.save(base_model.state_dict(), osp.join(cfg.output_dir, "target" + cfg.savename + ".pt"))

    return base_model


def obtain_label(loader, base_model, proj_head, cfg):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            inputs_test, labels, _ = next(iter_test)
            weak_x = inputs_test[1].cuda()

            if 'image' in cfg.SETTING.DATASET:
                weak_feas = base_model.netF(weak_x)
                if 'k' in cfg.SETTING.DATASET:
                    weak_outputs = base_model.netC(weak_feas)
                else:
                    weak_outputs = base_model.masking_layer(base_model.netC(weak_feas))

            else:
                weak_feas = base_model.encoder(weak_x)
                weak_outputs = base_model.fc(weak_feas)

            if start_test:
                all_feas = weak_feas.float().cpu()
                all_output = weak_outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_feas = torch.cat((all_feas, weak_feas.float().cpu()), 0)
                all_output = torch.cat((all_output, weak_outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    k = 3
    metric_tensor = proj_head.metric_tensor @ proj_head.metric_tensor.T
    metric_tensor = nn.Softmax(dim=1)(metric_tensor)
    metric_tensor = metric_tensor.float().cpu()
    start_test = True
    for i in range(len(all_feas)):
        v = all_feas[i] - all_feas
        metric_dist = torch.einsum('bi,ii,ib->b', v, metric_tensor, v.T)

        top_k_metric = torch.topk(- metric_dist, k).indices.tolist()

        top_k_outputs = all_output[top_k_metric]
        top_k_outputs = nn.Softmax(dim=1)(top_k_outputs)

        mixed_preds = top_k_outputs.mean(dim=0).unsqueeze(0)
        if start_test:
            all_mixed_preds = mixed_preds.float().cpu()
            start_test = False
        else:
            all_mixed_preds = torch.cat((all_mixed_preds, mixed_preds.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, all_output_pred = torch.max(all_output, dim=1)
    _, all_mix_output_pred = torch.max(all_mixed_preds, dim=1)

    accuracy = torch.sum(torch.squeeze(all_output_pred).float() == all_label).item() / float(all_label.size()[0])
    acc = torch.sum(torch.squeeze(all_mix_output_pred).float() == all_label).item() / float(all_label.size()[0])
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    logging.info(log_str)

    return all_mixed_preds
