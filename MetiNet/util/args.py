import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.optim

"""
    Utility functions for handling parsed arguments
"""


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Train a MetiNet')
    parser.add_argument('--dataset',
                        type=str,
                        default='CUB-200-2011',
                        help='Data set on MetiNet should be trained')
    parser.add_argument('--net',
                        type=str,
                        default='convnext_tiny_26',
                        help='Base network used as backbone of MetiNet. Default is convnext_tiny_26 with adapted strides to output 26x26 latent representations. Other option is convnext_tiny_13 that outputs 13x13 (smaller and faster to train, less fine-grained). Pretrained network on iNaturalist is only available for resnet50_inat. Options are: resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, convnext_tiny_26 and convnext_tiny_13.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size when training the model using minibatch gradient descent. Batch size is multiplied with number of available GPUs.')
    parser.add_argument('--epochs',
                        type=int,
                        default=60,
                        help='The number of epochs MetiNet should be trained')
    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        help='The optimizer that should be used for training')
    parser.add_argument('--lr_class',
                        type=float,
                        default=0.05,
                        help='The optimizer learning rate for training the weights from prototypes to classes (if using classification layer)')
    parser.add_argument('--lr_net',
                        type=float,
                        default=0.0005,
                        help='The optimizer learning rate for the backbone')
    parser.add_argument('--lr_color',
                        type=float,
                        default=0.0005,
                        help='The optimizer learning rate for the color network')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay used in the optimizer')
    parser.add_argument('--part_weight',
                        type=float,
                        default=1.0,
                        help='Weight of semantic part alignment loss')
    parser.add_argument('--proto_class_weight',
                        type=float,
                        default=1.0,
                        help='Weight of prototype classification loss')
    parser.add_argument('--color_class_weight',
                        type=float,
                        default=1.0,
                        help='Weight of color classification loss')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./runs/run_metinet',
                        help='The directory in which train progress should be logged')
    parser.add_argument('--num_parts',
                        type=int,
                        default=8,
                        help='Number of semantic parts')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Input images will be resized to --image_size x --image_size (square). Code only tested with 224x224, so no guarantees that it works for different sizes.')
    parser.add_argument('--state_dict_dir_net',
                        type=str,
                        default='',
                        help='The directory containing a state dict with a pretrained MEtiNet. E.g., ./runs/run_metinet/checkpoints/net_pretrained')
    parser.add_argument('--freeze_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs where pretrained backbone features will be frozen while training the rest'
                        )
    parser.add_argument('--dir_for_saving_images',
                        type=str,
                        default='visualization_results',
                        help='Directory for saving the prototypes and explanations')
    parser.add_argument('--disable_pretrained',
                        action='store_true',
                        help='When set, the backbone network is initialized with random weights instead of being pretrained on another dataset'
                        )
    parser.add_argument('--weighted_loss',
                        action='store_true',
                        help='Flag that weights the loss based on the class balance of the dataset. Recommended to use when data is imbalanced.')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed. Note that there will still be differences between runs due to nondeterminism. See https://pytorch.org/docs/stable/notes/randomness.html')
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='',
                        help='ID of gpu. Can be separated with comma')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Num workers in dataloaders')
    parser.add_argument('--bias',
                        action='store_true',
                        help='Flag that indicates whether to include a trainable bias in the linear classification layer'
                        )
    parser.add_argument('--use_classification_layer',
                        action='store_true',
                        help='Flag that indicates whether to use a linear classification layer')
    parser.add_argument('--num_classes',
                        type=int,
                        required=True,
                        help='Num classes in dataset')
    parser.add_argument('--aggregate',
                        type=str,
                        required=True,
                        help='How to aggregate base and color predictions. Options are: mean, product')
    parser.add_argument('--no_color_epochs',
                        type=int,
                        required=True,
                        help='Number of epochs until color network starts training')
    parser.add_argument('--strong_hue_augmentation',
                        type=float,
                        default=0.0,
                        help='Hue value for color jitter transform')
    parser.add_argument('--crop_augmentation',
                        type=float,
                        default=1.0,
                        help='Scale value for random resized crop transform')
    parser.add_argument('--fold',
                        type=int,
                        default=1,
                        help='Fold id for DIBAS dataset')

    args = parser.parse_args()
    if len(args.log_dir.split('/')) > 2:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

    assert not (args.strong_hue_augmentation != 0.0 and args.crop_augmentation != 1.0)

    return args


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)


def get_optimizer_nn(net, args: argparse.Namespace):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # create parameter groups
    params_to_train = []
    params_backbone = []

    # set up optimizer
    if 'resnet' in args.net:
        print("Chosen network is ResNet", flush=True)
        for name, param in net.module._net.named_parameters():
            # if 'layer4.2' or 'layer4.1' in name:
            if 'layer4.2' in name:
                params_to_train.append(param)
            elif 'layer4' in name or 'layer3' in name or 'layer2' in name:
                params_backbone.append(param)
            else:
                # param.requires_grad = False  # comment if enough memory
                params_backbone.append(param)  # comment if not enough memory

    elif 'convnext' in args.net:
        print("Chosen network is ConvNext", flush=True)
        for name, param in net.module._net.named_parameters():
            if 'features.7.2' in name:
                params_to_train.append(param)
            elif 'features.7' in name or 'features.6' in name or 'features.5' in name or 'features.4' in name:
                params_backbone.append(param)
            else:
                # param.requires_grad = False  # comment if enough memory
                params_backbone.append(param)  # comment if not enough memory
    else:
        print("Network is not ResNet or ConvNext", flush=True)

    classification_weight = []
    classification_bias = []
    for name, param in net.module._classification.named_parameters():
        if 'weight' in name:
            classification_weight.append(param)
        else:
            if args.bias:
                classification_bias.append(param)

    for param in net.module._add_on.parameters():
        params_to_train.append(param)

    params_color = []
    for param in net.module._color_net.parameters():
        params_color.append(param)

    paramlist_net = [
        {"params": params_backbone, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
        {"params": params_to_train, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
    ]

    paramlist_classifier = [
        {"params": classification_weight, "lr": args.lr_class, "weight_decay_rate": args.weight_decay},
        {"params": classification_bias, "lr": args.lr_class, "weight_decay_rate": 0},
    ]

    paramlist_color = [
        {"params": params_color, "lr": args.lr_color, "weight_decay_rate": args.weight_decay}
    ]

    if args.optimizer == 'Adam':
        # lr is overridden by lrs specified for each group
        optimizer_net = torch.optim.AdamW(paramlist_net, lr=args.lr_net, weight_decay=args.weight_decay)
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier, lr=args.lr_class, weight_decay=args.weight_decay)
        optimizer_color = torch.optim.AdamW(paramlist_color, lr=args.lr_color, weight_decay=args.weight_decay)
        return optimizer_net, optimizer_classifier, optimizer_color, params_to_train, params_backbone
    else:
        raise ValueError("This optimizer type is not implemented")
