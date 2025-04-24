import os.path

import numpy as np
import argparse
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from typing import Tuple, Dict
from torch import Tensor
import random
from torchvision.transforms.functional import to_tensor, to_grayscale
from torchvision.datasets import ImageFolder

from util.func import random_hue_jitter


def get_data(args: argparse.Namespace, strong_hue_augmentation: float, crop_augmentation: float):
    """
    Load the proper dataset based on the parsed arguments
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.dataset == 'CUB':
        return create_datasets('/shared/sets/datasets/lucid/cub/CUB_200_2011/dataset/train_crop',
                               '/shared/sets/datasets/lucid/cub/CUB_200_2011/dataset/train',
                               '/shared/sets/datasets/lucid/cub/CUB_200_2011/dataset/test_crop',
                               f'/shared/sets/datasets/lucid/cub_{args.num_parts}parts/train',
                               f'/shared/sets/datasets/lucid/cub_{args.num_parts}parts/test',
                               args.image_size,
                               args.num_parts,
                               8,
                               strong_hue_augmentation,
                               crop_augmentation)
    if args.dataset == 'CARS':
        return create_datasets('/shared/sets/datasets/lucid/cars/train',
                               '/shared/sets/datasets/lucid/cars/train',
                               '/shared/sets/datasets/lucid/cars/test',
                               f'/shared/sets/datasets/lucid/cars_{args.num_parts}parts/train',
                               f'/shared/sets/datasets/lucid/cars_{args.num_parts}parts/test',
                               args.image_size,
                               args.num_parts,
                               8,  # 32
                               strong_hue_augmentation,
                               crop_augmentation)
    if args.dataset == 'DOGS':
        return create_datasets('/shared/sets/datasets/lucid/dogs/train',
                               '/shared/sets/datasets/lucid/dogs/train',
                               '/shared/sets/datasets/lucid/dogs/test',
                               f'/shared/sets/datasets/lucid/dogs_{args.num_parts}parts/train',
                               f'/shared/sets/datasets/lucid/dogs_{args.num_parts}parts/test',
                               args.image_size,
                               args.num_parts,
                               8,  # 32
                               strong_hue_augmentation,
                               crop_augmentation)
    if args.dataset == 'FLOWERS':
        return create_datasets('/shared/sets/datasets/lucid/flowers/train',
                               '/shared/sets/datasets/lucid/flowers/train',
                               '/shared/sets/datasets/lucid/flowers/test',
                               f'/shared/sets/datasets/lucid/flowers_{args.num_parts}parts/train',
                               f'/shared/sets/datasets/lucid/flowers_{args.num_parts}parts/test',
                               args.image_size,
                               args.num_parts,
                               8,  # 32
                               strong_hue_augmentation)
    raise Exception(f'Could not load data set, data set "{args.dataset}" not found!')


def get_dataloaders(args: argparse.Namespace, strong_hue_augmentation: float, crop_augmentation: float):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, projectset, testset, classes, train_indices, targets = get_data(args, strong_hue_augmentation,
                                                                              crop_augmentation)

    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    to_shuffle = True
    sampler = None

    num_workers = args.num_workers

    if args.weighted_loss:
        if targets is None:
            raise ValueError("Weighted loss not implemented for this dataset. Targets should be restructured")
        # https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907
        class_sample_count = torch.tensor(
            [(targets[train_indices] == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        print("Weights for weighted sampler: ", weight, flush=True)
        samples_weight = torch.tensor([weight[t] for t in targets[train_indices]])
        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        to_shuffle = False

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=to_shuffle,
                                              sampler=sampler,
                                              pin_memory=cuda,
                                              num_workers=num_workers,
                                              worker_init_fn=np.random.seed(args.seed),
                                              drop_last=True)
    projectloader = torch.utils.data.DataLoader(projectset,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=cuda,
                                                num_workers=num_workers,
                                                worker_init_fn=np.random.seed(args.seed),
                                                drop_last=False)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False)

    print("Num classes (k) = ", len(classes), classes[:5], "etc.", flush=True)
    return trainloader, projectloader, testloader, classes


def create_datasets(train_dir: str, project_dir: str, test_dir: str, maps_train_dir: str, maps_test_dir: str,
                    img_size: int, num_parts: int, augment_shape_zoom: int, strong_hue_augmentation: float,
                    crop_augmentation: float):
    # Load raw datasets
    raw_trainset = ImageFolder(train_dir)
    raw_projectset = ImageFolder(project_dir)
    raw_testset = ImageFolder(test_dir)

    classes = raw_trainset.classes
    targets = raw_trainset.targets
    train_indices = list(range(len(raw_trainset)))

    if maps_train_dir != "":
        raw_maps_trainsets = [ImageFolder(os.path.join(maps_train_dir, str(i))) for i in range(num_parts)]
    else:
        raw_maps_trainsets = "dummy"
    if maps_test_dir != "":
        raw_maps_testsets = [ImageFolder(os.path.join(maps_test_dir, str(i))) for i in range(num_parts)]
    else:
        raw_maps_testsets = "dummy"

    # Define augmentations and transforms
    resize = transforms.Resize(size=(img_size, img_size))
    augment_shape = transforms.Compose([
        transforms.Resize(size=(img_size + augment_shape_zoom, img_size + augment_shape_zoom)),
        TrivialAugmentWideNoColor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.95, 1.)),
    ])
    # transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.95, 1.)),

    augment_hue_strongly = transforms.Lambda(lambda x: random_hue_jitter(x, strong_hue_augmentation))
    augment_crop = transforms.RandomResizedCrop(size=(img_size, img_size),
                                                scale=(crop_augmentation, crop_augmentation),
                                                ratio=(1.0, 1.0))

    grayscale_and_normalize = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.445, 0.445, 0.445), std=(0.269, 0.269, 0.269))
    ])
    # augment_hue_strongly already returns tensor
    grayscale_and_normalize_hue_aug = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Normalize(mean=(0.445, 0.445, 0.445), std=(0.269, 0.269, 0.269))
    ])

    # Load proper datasets
    if strong_hue_augmentation != 0.0:
        print(f'Strong hue augmentation switched on ({strong_hue_augmentation})')
        trainset = DatasetWithParts(raw_trainset, raw_maps_trainsets,
                                    augment_shape, augment_hue_strongly, grayscale_and_normalize_hue_aug)
        testset = DatasetWithParts(raw_testset, raw_maps_testsets,
                                   resize, augment_hue_strongly, grayscale_and_normalize_hue_aug)
    elif crop_augmentation != 1.0:
        print(f'Crop augmentation switched on ({crop_augmentation})')
        trainset = DatasetWithParts(raw_trainset, raw_maps_trainsets,
                                    augment_shape, augment_crop, grayscale_and_normalize)
        testset = DatasetWithParts(raw_testset, raw_maps_testsets,
                                   resize, augment_crop, grayscale_and_normalize)
    else:
        print('Additional augmentations switched off')
        trainset = DatasetWithParts(raw_trainset, raw_maps_trainsets,
                                    augment_shape, None, grayscale_and_normalize)
        testset = DatasetWithParts(raw_testset, raw_maps_testsets,
                                   resize, None, grayscale_and_normalize)

    projectset = DatasetWithParts(raw_projectset, None,
                                  resize, None, grayscale_and_normalize)

    return trainset, projectset, testset, classes, train_indices, torch.LongTensor(targets)


class DatasetWithParts(torch.utils.data.Dataset):
    """
    Returns:
        - Image after transform1 + transform2 + toTensor,
        - Image after transform1 + transform2 + transform3,
        - Maps after transform1 + toTensor,
        - Labels.
    """

    def __init__(self, dataset, maps_datasets, transform1, transform2, transform3):
        self.dataset = dataset
        self.maps_datasets = maps_datasets
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

        self.classes = dataset.classes
        self.imgs = dataset.imgs
        self.targets = dataset.targets

    def __getitem__(self, index):
        image, target = self.dataset[index]

        state = torch.get_rng_state()
        image = self.transform1(image)

        if self.maps_datasets is not None and self.maps_datasets != "dummy":
            maps = []
            for maps_dataset in self.maps_datasets:
                single_map, _ = maps_dataset[index]
                torch.set_rng_state(state)
                single_map = self.transform1(single_map)
                maps.append(to_tensor(to_grayscale(single_map)))
            maps = torch.stack(maps, 0).squeeze(1)
        else:
            maps = None

        if self.transform2:
            image1 = self.transform2(image)
            image2 = self.transform3(self.transform2(image))
        else:
            image1 = image
            image2 = self.transform3(image)

        if not torch.is_tensor(image1):
            image1 = to_tensor(image1)

        if self.maps_datasets == "dummy":
            return image1, image2, torch.tensor([1]), target
        elif self.maps_datasets:
            return image1, image2, maps, target
        else:
            return image1, image2, target

    def __len__(self):
        return len(self.dataset)


# function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True),
        }


class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {

            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
