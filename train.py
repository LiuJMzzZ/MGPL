import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import get_dataset
from data.splits import get_splits
from model import MGPLnet, train, val
from utils import setup_seed, weight_init
from utils import reset_prototypes



datasets = [
    'svhn',
    'cifar10',
    'cifar_plus_10',
    'cifar_plus_50',
    'tiny_imagenet'
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', default='cifar10', help='dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='initial_learning_rate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
    parser.add_argument('--h', type=int, default=128, help='dimension of the hidden layer')
    parser.add_argument('--k', type=int, default=3, help='number of subclusters in each class')
    parser.add_argument('--c', type=int, default=3, help='image channel')
    parser.add_argument('--temp_inter', type=float, default=0.1, help='temperature factor')
    parser.add_argument('--temp_intra', type=float, default=1, help='temperature factor')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--arch', default='resnet18', help='net arch')
    parser.add_argument('--split', type=int, default=0, help='unknown splits')
    parser.add_argument('--lamda', type=float, default=0.005, help='balance param between gen & dis')
    parser.add_argument('--clip', default=False, action='store_true', help='clip grad')


    args, _ = parser.parse_known_args()
    setup_seed(2022)
    os.environ["CUDA_VISIBLE_DEVICES"] = '%s' %args.gpu
    if not os.path.exists('./save_model/'):
        os.makedirs('./save_model/')

    known_classes, unknown_classes, known_dataset, unknown_dataset = get_splits(args.dset, num_split=args.split)
    args.num_classes = len(known_classes)
    train_set = get_dataset(known_dataset, True, known_classes, 'reindex')
    val_set = get_dataset(known_dataset, False, known_classes, 'reindex')
    open_set = get_dataset(unknown_dataset, False, unknown_classes, 'open')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False, num_workers=4)
    open_loader = DataLoader(open_set, batch_size=1000, shuffle=False, num_workers=4)
    
    mgplnet = MGPLnet(args.arch, args.c, args.h, args.k, args.num_classes, args.temp_inter, args.temp_intra)
    mgplnet = mgplnet.cuda()
    mgplnet.apply(weight_init)

    if args.dset == 'tiny_imagenet':
        args.clip = True
    optimizer = optim.Adam(mgplnet.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    print(args)

    best = 0
    for epoch in range(args.epoch):
        train(mgplnet, args, train_loader, epoch, optimizer)
        if epoch in [50, 80]: 
            mgplnet.prototypes = reset_prototypes(mgplnet, train_loader)
        val_acc = val(mgplnet, args, val_loader, epoch)
        scheduler.step()

        # save model
        if val_acc > best:
            torch.save(mgplnet, './save_model/{}_split_{}.pt'.format(args.dset, args.split))
            best = val_acc
        
    
    print('Finished')

