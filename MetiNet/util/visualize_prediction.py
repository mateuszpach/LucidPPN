import os, shutil
import argparse
from PIL import Image, ImageDraw as D
import torchvision
from util.func import get_patch_size
from torchvision import transforms
import torch
from util.vis_metinet import get_img_coordinates
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

try:
    import cv2
    use_opencv = True
except ImportError:
    use_opencv = False
    print("Heatmaps showing where a prototype is found will not be generated because OpenCV is not installed.", flush=True)


def vis_pred(net, vis_test_loader, classes, device, args: argparse.Namespace):
    random.seed(args.seed)

    # Make sure the model is in evaluation mode
    net.eval()

    save_dir = os.path.join(args.log_dir, args.dir_for_saving_images)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    patchsize, skip = get_patch_size(args)

    num_workers = args.num_workers

    imgs = vis_test_loader.dataset.imgs

    img_iter = tqdm(enumerate(vis_test_loader),
                    total=len(vis_test_loader),
                    mininterval=50.,
                    desc='Generating explanations...',
                    ncols=0)

    last_y = -1

    bbox_colors = ['#ff9900', '#0000ff', '#00ff00', '#ff00ff']
    global_results = []


    for k, (xs, x_augs, _, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        if ys[0] != last_y:
            last_y = ys[0]
            count_per_y = 0
        else:
            count_per_y +=1
            if count_per_y>1: #show max 5 imgs per class to speed up the process
                # continue
                pass
        xs, x_augs, ys = xs.to(device), x_augs.to(device), ys.to(device)

        img = imgs[k][0]
        img_name = os.path.splitext(os.path.basename(img))[0]
        dir = os.path.join(save_dir, img_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
            # shutil.copy(img, dir)

        p_scores = []
        c_scores = []
        to_skip = False

        with torch.no_grad():
            output = net(xs, x_augs, None, args.use_classification_layer)
            (grouped_proto_features, grouped_proto_pooled, grouped_color_features,
             grouped_color_pooled, agg, out) = output

            sorted_out, sorted_out_indices = torch.sort(out[0], descending=True)
            if sorted_out_indices[0] != ys[0]:
                # global_results.append(','.join([img_name, '-1', '-1', '-1', '-1', '-1']))
                continue

            shapetex_out = torch.mean(grouped_proto_pooled, dim=1)
            sorted_shapetex_out, sorted_shapetex_out_indices = torch.sort(shapetex_out[0], descending=True)

            if sorted_shapetex_out_indices[0] != ys[0]:
                print('Found!')
                # print('Shape')
                # for part_i in range(args.num_parts):
                #     proto_scores = grouped_proto_pooled[0, part_i, :]
                #     color_scores = grouped_color_pooled[0, part_i, :]
                #
                #     print(proto_scores[sorted_shapetex_out_indices[0]].item())
                #     # print(color_scores[sorted_shapetex_out_indices[0]].item())
                # print('Color')
                # for part_i in range(args.num_parts):
                #     proto_scores = grouped_proto_pooled[0, part_i, :]
                #     color_scores = grouped_color_pooled[0, part_i, :]
                #
                #     print(proto_scores[sorted_out_indices[0]].item())
                #     # print(color_scores[sorted_out_indices[0]].item())
            else:
                continue

            # # Keep only the ones where shaptexnet switched the top 2 candidates
            # if sorted_shapetex_out_indices[1] != sorted_out_indices[0] or sorted_shapetex_out_indices[0] != sorted_out_indices[1]:
            #     continue

            # Keep only the ones where shaptexnet had the same top 2 candidates
            # if sorted_shapetex_out_indices[0] != sorted_out_indices[0] or sorted_shapetex_out_indices[1] != sorted_out_indices[1]:
            #     continue

            class_names = ['A', 'B']
            # random.shuffle(class_names)

            for i, pred_class_idx in enumerate(sorted_out_indices[:1]):
                save_path = os.path.join(dir, class_names[i])
                os.makedirs(save_path, exist_ok=True)

                image_original = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img))
                image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img))
                draw = D.Draw(image)

                for part_i in range(args.num_parts):
                    bs, _, _, fs, fs = grouped_color_features.shape
                    softmaxes = grouped_color_features[:, part_i, :, :, :]

                    max_h, max_idx_h = torch.max(softmaxes[0, pred_class_idx, :, :], dim=0)
                    max_w, max_idx_w = torch.max(max_h, dim=0)
                    max_idx_h = max_idx_h[max_idx_w].item()
                    max_idx_w = max_idx_w.item()

                    draw.rectangle([(max_idx_w * skip, max_idx_h * skip), (min(args.image_size, max_idx_w * skip + patchsize), min(args.image_size, max_idx_h * skip + patchsize))], outline=bbox_colors[part_i], width=2)

                    proto_scores = grouped_proto_pooled[0, part_i, :]
                    sorted_proto_scores, sorted_proto_scores_indices = torch.sort(proto_scores, descending=True)

                    color_scores = grouped_color_pooled[0, part_i, :]
                    sorted_color_scores, sorted_color_scores_indices = torch.sort(color_scores, descending=True)

                    # if sorted_color_scores_indices[0] != ys[0] or sorted_proto_scores_indices[0] != sorted_color_scores_indices[0] or sorted_proto_scores_indices[1] != sorted_color_scores_indices[1]:
                    #     to_skip = True

                    # p_scores.append(sorted_proto_scores[i])
                    # c_scores.append(sorted_color_scores[i])
                    p_scores.append(proto_scores[pred_class_idx])
                    c_scores.append(color_scores[pred_class_idx])

                image_original.resize((168, 168)).save(os.path.join(save_path, 'original.png'))
                # image.resize((168, 168)).save(os.path.join(save_path, 'bbox.png'))

            for i, pred_class_idx in enumerate(sorted_shapetex_out_indices[:1]):
                save_path = os.path.join(dir, class_names[i])
                os.makedirs(save_path, exist_ok=True)

                for part_i in range(args.num_parts):
                    proto_scores = grouped_proto_pooled[0, part_i, :]
                    color_scores = grouped_color_pooled[0, part_i, :]

                    p_scores.append(proto_scores[pred_class_idx])
                    c_scores.append(color_scores[pred_class_idx])


        with open(os.path.join(dir, 'results.txt'), 'w') as f:
            f.write(','.join([class_names[0], str(sorted_out[0].item()), str(sorted_out[1].item())]))

        # global_results.append(','.join([img_name,
        #                                 class_names[0],
        #                                 str(sorted_out[0].item()),
        #                                 str(sorted_out[1].item()),
        #                                 str(sorted_out_indices[0].item()),
        #                                 str(sorted_out_indices[1].item())]))
        # When adding prototypes remember to switch in case class_name is B

        if p_scores[0].item() >= p_scores[4].item():
            to_skip = True
        if p_scores[1].item() >= p_scores[5].item():
            to_skip = True
        if p_scores[2].item() >= p_scores[6].item():
            to_skip = True
        if p_scores[3].item() >= p_scores[7].item():
            to_skip = True

        if not to_skip:
            global_results.append(','.join([img_name,
                                            class_names[0],
                                            str(sorted_out_indices[0].item()),
                                            str(sorted_out_indices[1].item()),
                                            str(p_scores[0].item()),
                                            str(p_scores[1].item()),
                                            str(p_scores[2].item()),
                                            str(p_scores[3].item()),
                                            str(p_scores[4].item()),
                                            str(p_scores[5].item()),
                                            str(p_scores[6].item()),
                                            str(p_scores[7].item()),
                                            str(c_scores[0].item()),
                                            str(c_scores[1].item()),
                                            str(c_scores[2].item()),
                                            str(c_scores[3].item()),
                                            str(c_scores[4].item()),
                                            str(c_scores[5].item()),
                                            str(c_scores[6].item()),
                                            str(c_scores[7].item()),
                                            ]))

    with open(os.path.join(save_dir, 'global_results.txt'), 'w') as f:
        f.write('\n'.join(global_results))

def vis_pred_experiments(net, imgs_dir, classes, device, args: argparse.Namespace):
    # Make sure the model is in evaluation mode
    net.eval()

    save_dir = os.path.join(os.path.join(args.log_dir, args.dir_for_saving_images),"Experiments")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    patchsize, skip = get_patch_size(args)

    num_workers = args.num_workers

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(args.image_size, args.image_size)),
                            transforms.ToTensor(),
                            normalize])

    vis_test_set = torchvision.datasets.ImageFolder(imgs_dir, transform=transform_no_augment)
    vis_test_loader = torch.utils.data.DataLoader(vis_test_set, batch_size = 1,
                                                shuffle=False, pin_memory=not args.disable_cuda and torch.cuda.is_available(),
                                                num_workers=num_workers)
    imgs = vis_test_set.imgs
    for k, (xs, ys) in enumerate(vis_test_loader): #shuffle is false so should lead to same order as in imgs
        
        xs, ys = xs.to(device), ys.to(device)
        img = imgs[k][0]
        img_name = os.path.splitext(os.path.basename(img))[0]
        dir = os.path.join(save_dir,img_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
            shutil.copy(img, dir)
        
        with torch.no_grad():
            softmaxes, pooled, out = net(xs, inference=True) #softmaxes has shape (bs, num_prototypes, W, H), pooled has shape (bs, num_prototypes), out has shape (bs, num_classes)
            sorted_out, sorted_out_indices = torch.sort(out.squeeze(0), descending=True)
            
            for pred_class_idx in sorted_out_indices:
                pred_class = classes[pred_class_idx]
                save_path = os.path.join(dir, str(f"{out[0,pred_class_idx].item():.3f}")+"_"+pred_class)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                sorted_pooled, sorted_pooled_indices = torch.sort(pooled.squeeze(0), descending=True)
                
                simweights = []
                for prototype_idx in sorted_pooled_indices:
                    simweight = pooled[0,prototype_idx].item() * net.module._classification.weight[pred_class_idx, prototype_idx].item()
                    
                    simweights.append(simweight)
                    if abs(simweight) > 0.01:
                        max_h, max_idx_h = torch.max(softmaxes[0, prototype_idx, :, :], dim=0)
                        max_w, max_idx_w = torch.max(max_h, dim=0)
                        max_idx_h = max_idx_h[max_idx_w].item()
                        max_idx_w = max_idx_w.item()
                        
                        image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img).convert("RGB"))
                        img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                        h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, max_idx_h, max_idx_w)
                        img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                        img_patch = transforms.ToPILImage()(img_tensor_patch)
                        img_patch.save(os.path.join(save_path, 'mul%s_p%s_sim%s_w%s_patch.png'%(str(f"{simweight:.3f}"),str(prototype_idx.item()),str(f"{pooled[0,prototype_idx].item():.3f}"),str(f"{net.module._classification.weight[pred_class_idx, prototype_idx].item():.3f}"))))
                        draw = D.Draw(image)
                        draw.rectangle([(max_idx_w*skip,max_idx_h*skip), (min(args.image_size, max_idx_w*skip+patchsize), min(args.image_size, max_idx_h*skip+patchsize))], outline='yellow', width=2)
                        image.save(os.path.join(save_path, 'mul%s_p%s_sim%s_w%s_rect.png'%(str(f"{simweight:.3f}"),str(prototype_idx.item()),str(f"{pooled[0,prototype_idx].item():.3f}"),str(f"{net.module._classification.weight[pred_class_idx, prototype_idx].item():.3f}"))))

