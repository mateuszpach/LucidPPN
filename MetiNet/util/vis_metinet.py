import argparse
import os
import random

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw as D
from tqdm import tqdm
from util.func import get_patch_size
import wandb
import torch.nn.functional as F
from sklearn.manifold import TSNE
import numpy as np
from scipy.ndimage import gaussian_filter

@torch.no_grad()
def visualize_topk(net, projectloader, device, foldername, args: argparse.Namespace, k=10):
    print("Visualizing prototypes for topk...", flush=True)
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype_gray = dict()
    tensors_per_prototype_rgb = dict()
    tensors_per_prototype_color = dict()
    tensors_per_prototype_color_compact = dict()
    tensors_per_prototype_ensemble = dict()
    classes_per_prototype = dict()

    for p in range(args.num_classes * args.num_parts):
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p]=near_imgs_dir
        seen_max[p]=0.
        saved[p]=0
        saved_ys[p]=[]
        tensors_per_prototype_gray[p]=[]
        tensors_per_prototype_rgb[p]=[]
        tensors_per_prototype_color[p]=[]
        tensors_per_prototype_color_compact[p]=[]
        tensors_per_prototype_ensemble[p]=[]
        classes_per_prototype[p]=[]

    patchsize, skip = get_patch_size(args)

    imgs = projectloader.dataset.imgs

    # Make sure the model is in evaluation mode
    net.eval()

    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=50.,
                    desc='Collecting topk',
                    ncols=0)

    # Iterate through the data
    images_seen = 0
    topks = dict()
    # Iterate through the training set
    for i, (xs, x_augs, ys) in img_iter:
        images_seen+=1
        x_augs, ys = x_augs.to(device), ys.to(device)
        # if i > 50:
        #     break
        with torch.no_grad():
            # Use the model to classify this batch of input data
            # output = net(None, xs, args.use_classification_layer)
            output = net(None, x_augs, None, args.use_classification_layer)
            (grouped_proto_features, grouped_proto_pooled, grouped_color_features,
             grouped_color_pooled, agg, out) = output
            bs, _, _, fs, fs = grouped_proto_features.shape
            pfs = torch.reshape(grouped_proto_features, (bs, -1, fs, fs))
            pooled = torch.reshape(grouped_proto_pooled, (bs, -1))

            pooled = pooled.squeeze(0)
            pfs = pfs.squeeze(0)

            for p in range(pooled.shape[0]):
                if p not in topks.keys():
                    topks[p] = []
                if len(topks[p]) < k:
                    topks[p].append((i, pooled[p].item()))
                else:
                    topks[p] = sorted(topks[p], key=lambda tup: tup[1], reverse=True)
                    if topks[p][-1][1] < pooled[p].item():
                        topks[p][-1] = (i, pooled[p].item())
                    if topks[p][-1][1] == pooled[p].item():
                        # equal scores. randomly chose one (since dataset is not shuffled so latter images with same scores can now also get in topk).
                        replace_choice = random.choice([0, 1])
                        if replace_choice > 0:
                            topks[p][-1] = (i, pooled[p].item())

    alli = []
    for p in topks.keys():
        for idx, score in topks[p]:
            alli.append(idx)

    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=50.,
                    desc='Visualizing topk',
                    ncols=0)
    for i, (xs, x_augs, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        # if i > 20:
        #     break
        if i in alli:
            x_augs, ys = x_augs.to(device), ys.to(device)
            for p in topks.keys():
                for idx, score in topks[p]:
                    if idx == i:
                        # Use the model to classify this batch of input data
                        with torch.no_grad():
                            # output = net(None, xs, args.use_classification_layer)
                            output = net(None, x_augs, None, args.use_classification_layer)
                            (grouped_proto_features, grouped_proto_pooled,
                             grouped_color_features, grouped_color_pooled, agg, out) = output
                            bs, _, _, fs, fs = grouped_proto_features.shape
                            softmaxes = torch.reshape(grouped_proto_features, (bs, -1, fs, fs))
                            pooled = torch.reshape(grouped_proto_pooled, (bs, -1))

                            #softmaxes has shape (1, num_prototypes, W, H)
                            outmax = torch.amax(out,dim=1)[0] #shape ([1]) because batch size of projectloader is 1

                        # Take the max per prototype.
                        max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
                        max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
                        max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1) #shape (num_prototypes)

                        h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                        w_idx = max_idx_per_prototype_w[p]

                        h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, h_idx, w_idx)

                        img_tensor_rgb = xs[0].unsqueeze_(0)
                        img_tensor_patch_rgb = img_tensor_rgb[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]

                        img_tensor = transforms.Grayscale(3)(xs[0].unsqueeze_(0))
                        img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]

                        saved[p]+=1
                        tensors_per_prototype_gray[p].append(img_tensor_patch)
                        tensors_per_prototype_rgb[p].append(img_tensor_patch_rgb)
                        classes_per_prototype[p].append(ys[0].item())

    # Visualize ENSEMBLE
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=50.,
                    desc='Collecting topk',
                    ncols=0)

    # Iterate through the data
    images_seen = 0
    topks = dict()
    # Iterate through the training set
    for i, (xs, x_augs, ys) in img_iter:
        images_seen+=1
        xs, x_augs, ys = xs.to(device), x_augs.to(device), ys.to(device)

        with torch.no_grad():
            # Use the model to classify this batch of input data
            # output = net(None, xs, args.use_classification_layer)
            output = net(xs, x_augs, None, args.use_classification_layer)
            (grouped_proto_features, grouped_proto_pooled, grouped_color_features,
             grouped_color_pooled, agg, out) = output
            bs, _, _, fs, fs = grouped_color_features.shape
            pfs = torch.reshape(grouped_color_features, (bs, -1, fs, fs))
            pooled = torch.reshape(grouped_color_pooled, (bs, -1))

            pooled = pooled.squeeze(0)
            pfs = pfs.squeeze(0)

            for p in range(pooled.shape[0]):
                if p not in topks.keys():
                    topks[p] = []
                if len(topks[p]) < k:
                    topks[p].append((i, pooled[p].item()))
                else:
                    topks[p] = sorted(topks[p], key=lambda tup: tup[1], reverse=True)
                    if topks[p][-1][1] < pooled[p].item():
                        topks[p][-1] = (i, pooled[p].item())
                    if topks[p][-1][1] == pooled[p].item():
                        # equal scores. randomly chose one (since dataset is not shuffled so latter images with same scores can now also get in topk).
                        replace_choice = random.choice([0, 1])
                        if replace_choice > 0:
                            topks[p][-1] = (i, pooled[p].item())

    alli = []
    for p in topks.keys():
        for idx, score in topks[p]:
            alli.append(idx)

    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=50.,
                    desc='Visualizing topk',
                    ncols=0)
    for i, (xs, x_augs, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        # if i > 20:
        #     break
        if i in alli:
            xs, x_augs, ys = xs.to(device), x_augs.to(device), ys.to(device)
            for p in topks.keys():
                for idx, score in topks[p]:
                    if idx == i:
                        # Use the model to classify this batch of input data
                        with torch.no_grad():
                            # output = net(None, xs, args.use_classification_layer)
                            output = net(xs, x_augs, None, args.use_classification_layer)
                            (grouped_proto_features, grouped_proto_pooled,
                             grouped_color_features, grouped_color_pooled, agg, out) = output
                            bs, _, _, fs, fs = grouped_color_features.shape
                            softmaxes = torch.reshape(grouped_color_features, (bs, -1, fs, fs))
                            pooled = torch.reshape(grouped_color_pooled, (bs, -1))

                            #softmaxes has shape (1, num_prototypes, W, H)
                            outmax = torch.amax(out,dim=1)[0] #shape ([1]) because batch size of projectloader is 1

                        # Take the max per prototype.
                        max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
                        max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
                        max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1) #shape (num_prototypes)

                        h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                        w_idx = max_idx_per_prototype_w[p]

                        h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, h_idx, w_idx)

                        img_tensor_rgb = xs[0].unsqueeze_(0)
                        img_tensor_patch_rgb = img_tensor_rgb[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max].cpu()

                        saved[p]+=1
                        tensors_per_prototype_ensemble[p].append(img_tensor_patch_rgb)

    # Color visualization
    k_color = 30
    # cube_size = 6
    cube_size = 5
    num_subcubes = cube_size ** 3
    cube_colors = torch.zeros((num_subcubes, 3))
    index = 0
    for r in range(cube_size):
        for g in range(cube_size):
            for b in range(cube_size):
                cube_colors[index] = torch.tensor([r / (cube_size - 1), g / (cube_size - 1), b / (cube_size - 1)])
                index += 1
    cube_colors = torch.clamp(cube_colors, min=0, max=1).to(device=device)
    # normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # cube_colors_normalized = normalize(cube_colors.permute(1, 0).unsqueeze(2)).squeeze().permute(1, 0)
    cube_colors_normalized = cube_colors

    color_purity_round = 0.0
    color_purity_70 = 0.0

    for part_i in range(args.num_parts):
        # onehot = F.one_hot(torch.tensor([part_i], device=device), num_classes=args.num_parts)
        # onehots = onehot.repeat([num_subcubes, 1])
        # scores = net.module._color_net(torch.cat([cube_colors_normalized, onehots], dim=1))
        scores = net.module._color_net(cube_colors_normalized)
        scores = scores.reshape(num_subcubes, args.num_parts, args.num_classes)
        scores = scores[:, part_i, :]

        color_purity_round += torch.sum(torch.round(scores)).item()

        for class_i in range(args.num_classes):
            c_scores, indices = torch.sort(scores[:, class_i], descending=True)
            top_k_indices = indices[:]
            top_k_colors = cube_colors[top_k_indices, :]
            top_k_rgbs = torch.round(top_k_colors * 255).to(dtype=torch.int32)
            top_k_c_scores = c_scores[:]

            colors_to_vis = []
            mini_imgs = []
            sum_c_score = 0
            color_threshold = 0.7 * torch.sum(top_k_c_scores).item()
            for i , (rgb, c_score) in enumerate(zip(top_k_rgbs, top_k_c_scores)):
                rgb_tuple = tuple(rgb.tolist())
                c_score = c_score.item()

                if i >= 4 * (k_color - 1) or sum_c_score > color_threshold:
                    break
                sum_c_score += c_score
                color_purity_70 += 1

                colors_to_vis.append(rgb.tolist())

                mini_color_img = Image.new("RGB",
                                           (img_tensor_patch.shape[1] // 2, img_tensor_patch.shape[2] // 2),
                                           rgb_tuple)
                mini_color_tensor = transforms.ToTensor()(mini_color_img)
                mini_imgs.append(mini_color_tensor)

                if len(mini_imgs) == 4:
                    color_grid = torchvision.utils.make_grid(mini_imgs, nrow=2, padding=1, pad_value=1)
                    color_tensor = color_grid[:, 1:-2, 1:-2]
                    tensors_per_prototype_color[part_i * args.num_classes + class_i].append(color_tensor)
                    mini_imgs = []

            if mini_imgs:
                while len(mini_imgs) < 4:
                    mini_color_img = Image.new("RGB",
                                               (img_tensor_patch.shape[1] // 2, img_tensor_patch.shape[2] // 2),
                                               (255, 255, 255))
                    mini_color_tensor = transforms.ToTensor()(mini_color_img)
                    mini_imgs.append(mini_color_tensor)
                color_grid = torchvision.utils.make_grid(mini_imgs, nrow=2, padding=1, pad_value=1)
                color_tensor = color_grid[:, 1:-2, 1:-2]
                tensors_per_prototype_color[part_i * args.num_classes + class_i].append(color_tensor)

            while len(tensors_per_prototype_color[part_i * args.num_classes + class_i]) < k_color:
                color_img = Image.new("RGB",
                                      (img_tensor_patch.shape[1], img_tensor_patch.shape[2]),
                                      (255, 255, 255))
                color_tensor = transforms.ToTensor()(color_img)
                tensors_per_prototype_color[part_i * args.num_classes + class_i].append(color_tensor)

            # if len(colors_to_vis) >= 3:
            #     X = np.array(colors_to_vis)
            #     tsne = TSNE(n_components=2, random_state=42, perplexity=len(colors_to_vis)//2)
            #     X_embedded = tsne.fit_transform(X)[:, 0]
            #     indices = np.argsort(X_embedded, axis=0)
            #     X_sorted = X[indices]
            #     colors_to_vis = X_sorted.tolist()
            #
            # # tensors = torch.zeros((img_tensor_patch.shape[1], img_tensor_patch.shape[1] * len(colors_to_vis), 3))  # Initialize an RGB array
            # # for i, col in enumerate(colors_to_vis):
            # #     tensors[:, i * img_tensor_patch.shape[1]:(i + 1) * img_tensor_patch.shape[1]] = torch.tensor(col)
            # tensors = torch.zeros((1, 1 * len(colors_to_vis), 3))  # Initialize an RGB array
            # for i, col in enumerate(colors_to_vis):
            #     tensors[:, i * 1:(i + 1) * 1] = torch.tensor(col)
            # tensors = torch.permute(tensors, (2, 0, 1))
            # tensors = tensors / 255
            #
            # tensors = transforms.Resize((img_tensor_patch.shape[1], 3 * img_tensor_patch.shape[2]), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(tensors)
            # # tensors = torch.from_numpy(gaussian_filter(tensors.numpy(), sigma=1))
            # tensors = list(torch.split(tensors, img_tensor_patch.shape[1], 2))
            # tensors_per_prototype_color_compact[part_i * args.num_classes + class_i].extend(tensors)

    total = args.num_parts * args.num_classes * num_subcubes
    wandb.log({f'{foldername}_color_purity_round': color_purity_round / total})
    wandb.log({f'{foldername}_color_purity_70': color_purity_70 / total})

    k_compact_color = 2
    for part_i in range(args.num_parts):
        for class_i in range(args.num_classes):
            print(part_i * args.num_classes + class_i, args.num_classes * args.num_parts, flush=True)
            colors_to_vis = torch.cat(tensors_per_prototype_ensemble[part_i * args.num_classes + class_i], dim=1)
            colors_to_vis = colors_to_vis.flatten(1).permute(1, 0)
            X = colors_to_vis.numpy()
            np.random.shuffle(X)
            # print(X.shape)
            X = X[::64]
            # print(X.shape)
            tsne = TSNE(n_components=2, random_state=42, perplexity=20)
            X_embedded = tsne.fit_transform(X)[:, 0]
            indices = np.argsort(X_embedded, axis=0)
            X_sorted = X[indices]
            colors_to_vis = X_sorted.tolist()

            tensors = torch.zeros((1, 1 * len(colors_to_vis), 3))  # Initialize an RGB array
            for i, col in enumerate(colors_to_vis):
                tensors[:, i * 1:(i + 1) * 1] = torch.tensor(col)
            tensors = torch.permute(tensors, (2, 0, 1))

            tensors = transforms.Resize((img_tensor_patch.shape[1], k_compact_color * img_tensor_patch.shape[2]), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(tensors)
            tensors = list(torch.split(tensors, img_tensor_patch.shape[1], 2))
            tensors_per_prototype_color_compact[part_i * args.num_classes + class_i].extend(tensors)

    compactness = 5
    all_all_tensors_compact = []
    for part_i in range(args.num_parts):
        all_tensors = []
        all_classes = []
        all_tensors_compact = []
        correct = 0.
        for class_i in range(args.num_classes):
            p = part_i * args.num_classes + class_i
            if saved[p]>0:
                # add text next to each topk-grid, to easily see which prototype it is
                text = str(p)
                txtimage = Image.new("RGB", (img_tensor_patch.shape[1],img_tensor_patch.shape[2]), (255, 255, 255))
                draw = D.Draw(txtimage)
                draw.text((img_tensor_patch.shape[0]//2, img_tensor_patch.shape[1]//2), text, anchor='mm', fill="black")
                txttensor = transforms.ToTensor()(txtimage)

                emptyimage = Image.new("RGB", (img_tensor_patch.shape[1], img_tensor_patch.shape[2]), (255, 255, 255))
                emptytensor = transforms.ToTensor()(emptyimage)

                tensors = ([txttensor] +
                           tensors_per_prototype_gray[p] +
                           [emptytensor] +
                           tensors_per_prototype_color[p] +
                           [emptytensor] +
                           tensors_per_prototype_rgb[p])

                tensors_compact = (tensors_per_prototype_gray[p][:compactness] +
                                   [emptytensor] +
                                   tensors_per_prototype_color_compact[p] +
                                   [emptytensor] +
                                   tensors_per_prototype_ensemble[p][:compactness])

                # save top-k image patches in grid
                grid = torchvision.utils.make_grid(tensors, nrow=1+k+1+k_color+1+k, padding=1, pad_value=1)
                torchvision.utils.save_image(grid,os.path.join(dir,"grid_topk_%s.png"%(str(p))))

                grid_compact = torchvision.utils.make_grid(tensors_compact, nrow=compactness+1+k_compact_color+1+compactness, padding=1, pad_value=1)
                torchvision.utils.save_image(grid_compact,os.path.join(dir, f"grid_compact_topk_{p}_part{part_i}_class{class_i}.png"))
                if saved[p]>=k:
                    all_tensors+=tensors
                    all_tensors_compact+=tensors_compact
                    all_classes.append(classes_per_prototype[p])
                    correct += len([x for x in classes_per_prototype[p] if x == class_i])

        grid = torchvision.utils.make_grid(all_tensors, nrow=1+k+1+k_color+1+k, padding=1, pad_value=1)
        save_path = os.path.join(dir, "grid_topk_part%s.png"%(str(part_i)))
        torchvision.utils.save_image(grid, save_path)
        wandb.log({f'{foldername}/{part_i}': wandb.Image(save_path)})

        grid_compact = torchvision.utils.make_grid(all_tensors_compact, nrow=1+compactness+1+3+1+compactness, padding=1, pad_value=1)
        save_path = os.path.join(dir, "grid_compact_topk_part%s.png"%(str(part_i)))
        torchvision.utils.save_image(grid_compact, save_path)
        wandb.log({f'{foldername}_compact/{part_i}': wandb.Image(save_path)})


        classes_table = wandb.Table(columns=list(range(k)), data=all_classes)
        wandb.log({f'{foldername}_classes/{part_i}': classes_table})
        wandb.log({f'{foldername}_acc/{part_i}': correct / (args.num_classes * k)})

        all_all_tensors_compact += all_tensors_compact

    # for val, adj_type in [(0, 'none'), (1.25, 'contrast'), (1.5, 'contrast'), (1.5, 'gamma'), (2.2, 'gamma')]:
    for val, adj_type in [(0, 'none')]:
        for class_i in range(args.num_classes):
            nrow = compactness + 1 + k_compact_color + 1 + compactness
            all_tensors_compact_per_class = []
            for part_i in range(args.num_parts):
                l = (part_i * args.num_classes + class_i) * nrow
                r = l + nrow
                all_tensors_compact_per_class.extend(all_all_tensors_compact[l:r])
            grid_compact = torchvision.utils.make_grid(all_tensors_compact_per_class, nrow=nrow, padding=1, pad_value=1)

            chunked = []
            chunked.append(grid_compact[:, :, :(compactness + 1) * (1 + img_tensor_patch.shape[2])])
            for i in range(k_compact_color):
                l = (compactness + 1 + i) * (1 + img_tensor_patch.shape[2]) + 1
                r = (compactness + 1 + i + 1) * (1 + img_tensor_patch.shape[2])
                chunked.append(grid_compact[:, :, l:r])
            chunked.append(grid_compact[:, :, (compactness + 1 + k_compact_color) * (1 + img_tensor_patch.shape[2]):])
            grid_compact = torch.cat(chunked, dim=2)

            gap = 12
            chunked = []
            chunked.append(grid_compact[:, :(img_tensor_patch.shape[1] + 1), :])
            for i in range(1, args.num_parts):
                chunked.append(torch.ones((grid_compact.shape[0], gap, grid_compact.shape[2])))
                u = i * (img_tensor_patch.shape[1] + 1)
                l = (i + 1) * (img_tensor_patch.shape[1] + 1)
                chunked.append(grid_compact[:, u:l, :])
            grid_compact = torch.cat(chunked, dim=1)

            if adj_type == 'contrast':
                grid_compact = transforms.functional.adjust_contrast(grid_compact, val)
            if adj_type == 'gamma':
                grid_compact = transforms.functional.adjust_gamma(grid_compact, val)

            save_path = os.path.join(dir, f'{adj_type}{str(val)}', f'grid_compact_class{class_i}.png')
            os.makedirs(os.path.join(dir, f'{adj_type}{str(val)}'), exist_ok=True)
            torchvision.utils.save_image(grid_compact, save_path)
            if adj_type == 'none':
                wandb.log({f'{foldername}_compact_per_class/{class_i}': wandb.Image(save_path)})

    return topks


def visualize(net, projectloader, num_classes, device, foldername, args: argparse.Namespace):
    print("Visualizing prototypes...", flush=True)
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()
    abstainedimgs = set()
    notabstainedimgs = set()

    for p in range(args.num_classes * args.num_parts):
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p]=near_imgs_dir
        seen_max[p]=0.
        saved[p]=0
        saved_ys[p]=[]
        tensors_per_prototype[p]=[]

    patchsize, skip = get_patch_size(args)

    imgs = projectloader.dataset.imgs

    # skip some images for visualisation to speed up the process
    if len(imgs)/num_classes <10:
        skip_img=10
    elif len(imgs)/num_classes < 50:
        skip_img=5
    else:
        skip_img = 2
    print("Every", skip_img, "is skipped in order to speed up the visualisation process", flush=True)

    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=100.,
                    desc='Visualizing',
                    ncols=0)

    # Iterate through the data
    images_seen_before = 0
    for i, (xs, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        if i % skip_img == 0:
            images_seen_before+=xs.shape[0]
            continue

        xs, ys = xs.to(device), ys.to(device)
        # Use the model to classify this batch of input data
        with torch.no_grad():
            softmaxes, _, out = net(xs, inference=True)

        max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
        # In PyTorch, images are represented as [channels, height, width]
        max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
        max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1)
        for p in range(0, net.module._num_prototypes):
            c_weight = torch.max(classification_weights[:,p]) #ignore prototypes that are not relevant to any class
            if c_weight>0:
                h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                w_idx = max_idx_per_prototype_w[p]
                idx_to_select = max_idx_per_prototype[p,h_idx, w_idx].item()
                found_max = max_per_prototype[p,h_idx, w_idx].item()

                imgname = imgs[images_seen_before+idx_to_select]
                if out.max() < 1e-8:
                    abstainedimgs.add(imgname)
                else:
                    notabstainedimgs.add(imgname)

                if found_max > seen_max[p]:
                    seen_max[p]=found_max

                if found_max > 0.5:
                    img_to_open = imgs[images_seen_before+idx_to_select]
                    if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): #dataset contains tuples of (img,label)
                        imglabel = img_to_open[1]
                        img_to_open = img_to_open[0]

                    image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_to_open).convert("RGB"))
                    img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, h_idx, w_idx)
                    img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                    saved[p]+=1
                    tensors_per_prototype[p].append((img_tensor_patch, found_max))

                    save_path = os.path.join(dir, "prototype_%s")%str(p)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    draw = D.Draw(image)
                    draw.rectangle([(w_coor_min,h_coor_min), (w_coor_max, h_coor_max)], outline='yellow', width=2)
                    image.save(os.path.join(save_path, 'p%s_%s_%s_%s_rect.png'%(str(p),str(imglabel),str(round(found_max, 2)),str(img_to_open.split('/')[-1].split('.jpg')[0]))))


        images_seen_before+=len(ys)

    print("num images abstained: ", len(abstainedimgs), flush=True)
    print("num images not abstained: ", len(notabstainedimgs), flush=True)
    for p in range(net.module._num_prototypes):
        if saved[p]>0:
            try:
                sorted_by_second = sorted(tensors_per_prototype[p], key=lambda tup: tup[1], reverse=True)
                sorted_ps = [i[0] for i in sorted_by_second]
                grid = torchvision.utils.make_grid(sorted_ps, nrow=16, padding=1)
                torchvision.utils.save_image(grid,os.path.join(dir,"grid_%s.png"%(str(p))))
            except RuntimeError:
                pass

# convert latent location to coordinates of image patch
def get_img_coordinates(img_size, softmaxes_shape, patchsize, skip, h_idx, w_idx):
    # in case latent output size is 26x26. For convnext with smaller strides. 
    if softmaxes_shape[1] == 26 and softmaxes_shape[2] == 26:
        #Since the outer latent patches have a smaller receptive field, skip size is set to 4 for the first and last patch. 8 for rest.
        h_coor_min = max(0,(h_idx-1)*skip+4)
        if h_idx < softmaxes_shape[-1]-1:
            h_coor_max = h_coor_min + patchsize
        else:
            h_coor_min -= 4
            h_coor_max = h_coor_min + patchsize
        w_coor_min = max(0,(w_idx-1)*skip+4)
        if w_idx < softmaxes_shape[-1]-1:
            w_coor_max = w_coor_min + patchsize
        else:
            w_coor_min -= 4
            w_coor_max = w_coor_min + patchsize
    else:
        h_coor_min = h_idx*skip
        h_coor_max = min(img_size, h_idx*skip+patchsize)
        w_coor_min = w_idx*skip
        w_coor_max = min(img_size, w_idx*skip+patchsize)

    if h_idx == softmaxes_shape[1]-1:
        h_coor_max = img_size
    if w_idx == softmaxes_shape[2] -1:
        w_coor_max = img_size
    if h_coor_max == img_size:
        h_coor_min = img_size-patchsize
    if w_coor_max == img_size:
        w_coor_min = img_size-patchsize

    return h_coor_min, h_coor_max, w_coor_min, w_coor_max
