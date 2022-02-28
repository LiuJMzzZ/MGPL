import argparse
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from data.dataset import get_dataset
from data.splits import get_splits
from utils import  setup_seed


def get_output(model, data_loader):
    z = []
    dists = []
    labels = []
    kl_divs = []
    with torch.no_grad():
        for images, label in data_loader:
            images = images.cuda()
            z_iter, dist_iter, kl_div_iter, _ = model(images)
            z.extend(z_iter.cpu().data.numpy())
            dists.extend(dist_iter.cpu().data.numpy())
            kl_divs.extend(kl_div_iter.cpu().data.numpy())
            labels.extend(label.data.numpy())
    z = np.array(z)
    dists = np.array(dists)
    kl_divs = np.array(kl_divs)
    labels = np.array(labels)
    # dist
    dist_reshape = dists.reshape((len(dists), model.n_classes, model.n_sub_prototypes)) 
    dist_class_min = dist_reshape.min(2)  # min dist in each class
    dist_min = np.min(dist_class_min, 1)
    dist_pred = np.argmin(dist_class_min, 1)
    # kl_div
    kld_reshape = kl_divs.reshape((len(dists), model.n_classes, model.n_sub_prototypes)) 
    kld_class_min = kld_reshape.min(2)  # min kld in each class
    kld_min = np.min(kld_class_min, 1)
    kld_pred = np.argmin(kld_class_min, 1)

    return z, labels, dist_min, kld_min, dist_pred, kld_pred

def auroc_score(inner_score, open_score):  
    y_true = np.array([0] * len(inner_score) + [1] * len(open_score))
    y_score = np.concatenate([inner_score, open_score])
    auc_score = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
    opt_threshold = thresholds[maxindex]
    print('*'*50)
    print('Openset AUROC Score')
    print('avg known score: {:.03f}, avg unknown score: {:.03f}, AUROC score {:.03f}'.format(
        np.mean(inner_score), np.mean(open_score), auc_score))
    return auc_score

def inner_acc(pred_inner, labels_inner):
    inner_corrects = np.sum(pred_inner == labels_inner)
    inner_num = len(labels_inner)
    acc = inner_corrects / inner_num
    print('*'*50)
    print('Innerset Classification Performance')
    print('inner corrects: {} inner samples: {} inner accuracy {}'.format(
            inner_corrects, inner_num, inner_corrects / inner_num))
    return acc

def test_openset(model, inner_loader, open_loader):
    model.eval()
    model = model.cuda()
    _, inner_label, _, inner_score, _, inner_pred = get_output(model, inner_loader)
    _, open_label, _, open_score, _, open_pred = get_output(model, open_loader)
    acc = inner_acc(inner_pred, inner_label)
    auroc = auroc_score(inner_score, open_score)
    return acc, auroc


#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', default='cifar10', help='dataset')
    parser.add_argument('--split', type=int, default=0, help='unknown splits')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    
    datasets = [
    'svhn',
    'cifar10',
    'cifar_plus_10',
    'cifar_plus_50',
    'tiny_imagenet'
    ]

    setup_seed(2021)
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '%s' %args.gpu

    known_classes, unknown_classes, known_dataset, unknown_dataset = get_splits(args.dset, num_split=args.split)

    print('Unknown Detection Result')
    print('Dataset: {}    Split: {}'.format(args.dset, args.split))

    inner_set = get_dataset(known_dataset, False, known_classes, 'reindex')
    open_set = get_dataset(unknown_dataset, False, unknown_classes, 'open')
    inner_loader = DataLoader(inner_set, batch_size=1000, shuffle=False, num_workers=4)
    open_loader = DataLoader(open_set, batch_size=1000, shuffle=False, num_workers=4)
    
    model_dir = './save_model/{}_split_{}.pt'.format(args.dset, args.split)
    model = torch.load(model_dir)
    acc, auroc = test_openset(model, inner_loader, open_loader)










