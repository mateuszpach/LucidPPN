import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.utils.data
import torch.utils.data
import torchvision.io
import torchvision.transforms.v2 as transforms
from torchvision.models import resnet101, ResNet101_Weights

from datasets import CUBDataset
from nets import IndividualLandmarkNet
from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore", message="nn.functional.upsample_bilinear is deprecated. Use nn.functional.interpolate instead.")


class CUBDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, image_size: int = 224):
        self.data_path = data_path
        self.image_size = image_size

        self.class_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

        self.names = []
        for i, class_dir in enumerate(self.class_dirs):
            class_path = os.path.join(data_path, class_dir)
            # Get list of image files for each class
            images = [f for f in os.listdir(class_path) if f.endswith('.jpg') or f.endswith('.png')]
            self.names.extend([os.path.join(class_dir, img) for img in images])

        self.transform = transforms.Compose([
            transforms.Resize(size=image_size, antialias=True),
            transforms.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.names[idx])
        im = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.RGB)
        t_im = self.transform(im)
        return t_im, im.shape[-2:], self.names[idx]


class CarsDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, image_size: int = 224):
        self.data_path = data_path
        self.image_size = image_size

        self.class_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

        self.names = []
        for i, class_dir in enumerate(self.class_dirs):
            class_path = os.path.join(data_path, class_dir)
            # Get list of image files for each class
            images = [f for f in os.listdir(class_path) if f.endswith('.jpg') or f.endswith('.png')]
            self.names.extend([os.path.join(class_dir, img) for img in images])

        self.transform = transforms.Compose([
            transforms.Resize(size=(image_size, image_size), antialias=True),
            transforms.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.names[idx])
        im = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.RGB)
        t_im = self.transform(im)
        return t_im, im.shape[-2:], self.names[idx]


class DogsDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, image_size: int = 224):
        self.data_path = data_path
        self.image_size = image_size

        self.class_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

        self.names = []
        for i, class_dir in enumerate(self.class_dirs):
            class_path = os.path.join(data_path, class_dir)
            # Get list of image files for each class
            images = [f for f in os.listdir(class_path) if f.endswith('.jpg') or f.endswith('.png')]
            self.names.extend([os.path.join(class_dir, img) for img in images])

        self.transform = transforms.Compose([
            transforms.Resize(size=(image_size, image_size), antialias=True),
            transforms.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.names[idx])
        im = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.RGB)
        t_im = self.transform(im)
        return t_im, im.shape[-2:], self.names[idx]


class FlowersDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, image_size: int = 224):
        self.data_path = data_path
        self.image_size = image_size

        self.class_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

        self.names = []
        for i, class_dir in enumerate(self.class_dirs):
            class_path = os.path.join(data_path, class_dir)
            # Get list of image files for each class
            images = [f for f in os.listdir(class_path) if f.endswith('.jpg') or f.endswith('.png')]
            self.names.extend([os.path.join(class_dir, img) for img in images])

        self.transform = transforms.Compose([
            transforms.Resize(size=(image_size, image_size), antialias=True),
            transforms.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.names[idx])
        im = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.RGB)
        t_im = self.transform(im)
        return t_im, im.shape[-2:], self.names[idx]


def save_maps(loader, net, device, save_root):
    pbar = tqdm(loader, position=0, leave=True)
    with torch.no_grad():
        for i, (X, original_size, path_suffix) in enumerate(tqdm(loader)):
            _, maps, _ = net(X.to(device))
            resized_maps = transforms.functional.resize(maps, original_size, antialias=True)
            for j in range(maps.shape[1]):
                save_path = f'{save_root}/{j}/{path_suffix[0]}'
                directory = os.path.dirname(save_path)
                os.makedirs(directory, exist_ok=True)
                reversed_image_pil = transforms.functional.to_pil_image(resized_maps[0, j, :, :])
                reversed_image_pil.save(save_path)
    pbar.close()
def main():
    parser = argparse.ArgumentParser(description='PDiscoNet')
    parser.add_argument('--model_name', help='Name under which the model will be saved', required=True)
    parser.add_argument('--data_root', help='directory that contains the celeba, cub, or partimagenet folder',
                        required=True)
    parser.add_argument('--dataset', help='The dataset to use. Choose celeba, cub, or partimagenet.', required=True)
    parser.add_argument('--num_parts', help='number of parts to predict', default=8, type=int)
    parser.add_argument('--image_size', default=448, type=int)  # 256 for celeba, 448 for cub,  224 for partimagenet
    parser.add_argument('--pretrained_model_path', default='',
                        help='If you want to load a pretrained model specify the path to the model here.')
    parser.add_argument('--save_root', help='directory to save maps', required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = args.data_root + '/' + args.dataset.lower()
    if args.dataset.lower() == 'cub':
        dataset_train = CUBDataset(data_path + '/train_cropped', image_size=args.image_size)
        dataset_test = CUBDataset(data_path + '/test_cropped', image_size=args.image_size)
        num_cls = 200
    elif args.dataset.lower() == 'cars':
        dataset_train = CarsDataset(data_path + '/train', image_size=args.image_size)
        dataset_test = CarsDataset(data_path + '/test', image_size=args.image_size)
        num_cls = 196
    elif args.dataset.lower() == 'dogs':
        dataset_train = DogsDataset(data_path + '/train', image_size=args.image_size)
        dataset_test = DogsDataset(data_path + '/test', image_size=args.image_size)
        num_cls = 120
    elif args.dataset.lower() == 'flowers':
        dataset_train = FlowersDataset(data_path + '/train', image_size=args.image_size)
        dataset_test = FlowersDataset(data_path + '/test', image_size=args.image_size)
        num_cls = 102
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=4)

    weights = ResNet101_Weights.DEFAULT
    basenet = resnet101(weights=weights)
    net = IndividualLandmarkNet(basenet, args.num_parts, num_classes=num_cls)
    net.load_state_dict(torch.load(args.pretrained_model_path))
    net.to(device)
    net.eval()

    save_maps(train_loader, net, device, os.path.join(args.save_root, 'train'))
    save_maps(test_loader, net, device, os.path.join(args.save_root, 'test'))


if __name__ == "__main__":
    main()
