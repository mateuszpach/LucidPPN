from metinet.metinet import MetiNet, get_network
from util.log import Log
import torch.nn as nn
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders
from util.func import init_weights_xavier
from metinet.train import train_metinet
from metinet.test import eval_metinet
from metinet.final_test import final_eval_metinet
from util.eval_cub_csv import eval_prototypes_cub_parts_csv, get_topk_cub, get_proto_patches_cub
import torch
from util.vis_metinet import visualize, visualize_topk
from util.visualize_prediction import vis_pred, vis_pred_experiments
import sys, os
import random
import numpy as np
from shutil import copy
import matplotlib.pyplot as plt
from copy import deepcopy
import wandb


def run_metinet(args=None):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args = args or get_args()
    assert args.batch_size > 1

    wandb.init(
        project="unambiguous-prototypes",
        name=f"MetiNet-{args.dataset}-{args.num_parts}parts",
        entity="mateuszpach",
        config=args
    )

    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids != '':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))

    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids) == 1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
        elif len(device_ids) == 0:
            device = torch.device('cuda')
            print("CUDA device set without id specification", flush=True)
            device_ids.append(torch.cuda.current_device())
        else:
            print(
                "This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU.",
                flush=True)
            device_str = ''
            for d in device_ids:
                device_str += str(d)
                device_str += ","
            device = torch.device('cuda:' + str(device_ids[0]))
    else:
        device = torch.device('cpu')

    # Log which device was actually used
    print(f'Device used: {device} with id {device_ids}', flush=True)

    # Get the initial dataset and loaders
    trainloader, projectloader, testloader, classes = get_dataloaders(args, args.strong_hue_augmentation,
                                                                      args.crop_augmentation)

    # Create MetiNet subnetworks
    (feature_net, add_on_layers, pool_layer, classification_layer,
     color_net, num_prototypes) = get_network(len(classes), args)
    # Create the MetiNet
    net = MetiNet(num_classes=len(classes),
                  num_parts=args.num_parts,
                  feature_net=feature_net,
                  add_on_layers=add_on_layers,
                  pool_layer=pool_layer,
                  classification_layer=classification_layer,
                  color_net=color_net)

    wandb.watch(net)
    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids=device_ids)

    # Define loss functions
    part_criterion = nn.BCELoss(reduction='mean').to(device)
    class_criterion = nn.BCELoss(reduction='mean').to(device)

    # Define optimizers
    optimizer_net, optimizer_classifier, optimizer_color, params_to_train, params_backbone = get_optimizer_nn(net, args)

    # Define schedulers
    def scheduler_net_lr_lambda(current_epoch):
        if current_epoch > args.epochs:
            return 0
        elif current_epoch > args.freeze_epochs:
            return 0.1
        else:
            return 1

    scheduler_net = torch.optim.lr_scheduler.LambdaLR(optimizer_net, lr_lambda=scheduler_net_lr_lambda)

    def scheduler_color_lr_lambda(current_epoch):
        if current_epoch > args.no_color_epochs:
            return 1
        else:
            return 0

    scheduler_color = torch.optim.lr_scheduler.LambdaLR(optimizer_color, lr_lambda=scheduler_color_lr_lambda)

    scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier,
                                                                                T_0=10,
                                                                                eta_min=0.001,
                                                                                T_mult=1,
                                                                                verbose=False)

    # Initialize or load model
    with torch.no_grad():
        if args.state_dict_dir_net != '':
            print(f'Loading from checkpoint: {args.state_dict_dir_net}', flush=True)
            checkpoint = torch.load(args.state_dict_dir_net, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'], strict=True)
            # base_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if 'color' not in k}
            # net.load_state_dict(base_state_dict, strict=False)
            print('Network loaded', flush=True)
            optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict'])
            optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
            optimizer_color.load_state_dict(checkpoint['optimizer_color_state_dict'])
            print('Optimizers loaded', flush=True)
            scheduler_net.load_state_dict(checkpoint['scheduler_net_state_dict'])
            scheduler_classifier.load_state_dict(checkpoint['scheduler_classifier_state_dict'])
            scheduler_color.load_state_dict(checkpoint['scheduler_color_state_dict'])
            print('Schedulers loaded', flush=True)
            start_epoch = checkpoint['epoch'] + 1
            print(f'Starting from epoch: {start_epoch}', flush=True)
        else:
            net.module._add_on.apply(init_weights_xavier)
            torch.nn.init.normal_(net.module._classification.weight, mean=1.0, std=0.1)
            if args.bias:
                torch.nn.init.constant_(net.module._classification.bias, val=0.)
            print("Classification layer initialized with mean",
                  torch.mean(net.module._classification.weight).item(),
                  flush=True)
            start_epoch = 1

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        x, x_aug, m, y = next(iter(trainloader))
        x, x_aug = x.to(device), x_aug.to(device)
        proto_features, _, _, _, _, _ = net(x, x_aug, m)
        wshape = proto_features.shape[-1]
        args.wshape = wshape  # needed for calculating image patch size
        print(f'Output shape: {proto_features.shape}', flush=True)

    # Start by training only top layers
    backbone_frozen = True
    for param in net.module._classification.parameters():
        param.requires_grad = True
    for param in net.module._color_net.parameters():
        param.requires_grad = True
    for param in params_to_train:
        param.requires_grad = True
    for param in params_backbone:
        param.requires_grad = False

    # Start by not training the color net
    no_color = True

    for epoch in range(start_epoch, args.epochs + 1 + args.no_color_epochs):
        # Unfreeze the backbone after freeze_epochs
        if epoch > args.freeze_epochs and backbone_frozen:
            for param in params_backbone:
                param.requires_grad = True
            backbone_frozen = False

        # Enable learning of color network after no_color_epochs
        if epoch > args.no_color_epochs and no_color:
            no_color = False

        print(f'Epoch: {epoch}, '
              f'backbone_frozen: {backbone_frozen}, '
              f'no_color: {no_color}', flush=True)

        # Train for one epoch
        train_info = train_metinet(net, trainloader, optimizer_net, optimizer_classifier, optimizer_color,
                                   scheduler_net, scheduler_classifier, scheduler_color, no_color, part_criterion,
                                   class_criterion, epoch, device, args.use_classification_layer, args.part_weight,
                                   args.proto_class_weight, args.color_class_weight, args.num_classes, args.num_parts,
                                   args.aggregate)

        # Evaluate
        eval_info = eval_metinet(net, testloader, epoch, device, args.use_classification_layer, args.num_parts,
                                 args.aggregate, '', args.dataset)

        # Log metrics
        wandb.log({**train_info, **eval_info}, step=epoch - start_epoch + 1)

        # Save the trained network after critical epochs
        if epoch in [args.freeze_epochs, args.no_color_epochs, args.epochs]:
            net.eval()
            torch.save(obj={'model_state_dict': net.state_dict(),
                            'optimizer_net_state_dict': optimizer_net.state_dict(),
                            'optimizer_classifier_state_dict': optimizer_classifier.state_dict(),
                            'optimizer_color_state_dict': optimizer_color.state_dict(),
                            'scheduler_net_state_dict': scheduler_net.state_dict(),
                            'scheduler_classifier_state_dict': scheduler_classifier.state_dict(),
                            'scheduler_color_state_dict': scheduler_color.state_dict(),
                            'epoch': epoch},
                       f=os.path.join(args.log_dir, f'checkpoint_epoch{epoch}'))

    if start_epoch == args.epochs + 1 + args.no_color_epochs and False:
        eval_info = eval_metinet(net, testloader, 0, device, args.use_classification_layer, args.num_parts,
                                 args.aggregate, '', args.dataset)

        wandb.log(eval_info)
    # return
    # final_eval_info = final_eval_metinet(net, testloader, 0, device, args.use_classification_layer,
    #                                      args.num_classes, args.aggregate, '')
    #
    # # Log metrics
    # wandb.log(final_eval_info)

    # Visualize prototypes
    # topks = visualize_topk(net, projectloader, device, 'visualised_prototypes_topk', args)

    args_bs1 = deepcopy(args)
    args_bs1.batch_size = 1
    _, _, testloader_bs1, _ = get_dataloaders(args_bs1, args_bs1.strong_hue_augmentation, args_bs1.crop_augmentation)
    vis_pred(net, testloader_bs1, classes, device, args)

    # Report classifier weights
    if args.use_classification_layer:
        print("Classifier weights: ", net.module._classification.weight, flush=True)
        print("Classifier bias: ", net.module._classification.bias, flush=True)

    # Evaluate prototype purity
    # if args.dataset == 'CUB':
    #     project_path = '/home/z1164034/datasets/cub/CUB_200_2011/'
    #     parts_loc_path = os.path.join(project_path, "parts/part_locs.txt")
    #     parts_name_path = os.path.join(project_path, "parts/parts.txt")
    #     imgs_id_path = os.path.join(project_path, "images.txt")
    #
    #     net.eval()
    #     print("\n\nEvaluating cub prototypes for training set", flush=True)
    #     csvfile_topk = get_topk_cub(net, projectloader, 10, 'train', device, args)
    #     eval_prototypes_cub_parts_csv(csvfile_topk, parts_loc_path, parts_name_path, imgs_id_path,
    #                                   'train_topk', args, None)

    print("Done!", flush=True)


if __name__ == '__main__':
    args = get_args()

    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    run_metinet(args)
