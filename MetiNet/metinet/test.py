import torch
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.func import topk_accuracy
from torchvision.transforms.functional import resize


@torch.no_grad()
def eval_metinet(net,
                 test_loader: DataLoader,
                 epoch,
                 device,
                 use_classification_layer,
                 num_parts,
                 aggregate,
                 mode,
                 dataset,
                 progress_prefix: str = 'Eval Epoch') -> dict:
    net = net.to(device)

    # Make sure the model is in evaluation mode
    net.eval()

    # Keep an info dict about the procedure
    info = dict()

    global_top1acc = 0.
    global_top5acc = 0.

    global_top1acc_proto = 0.
    global_top1accs_proto = [0.] * num_parts
    global_top5acc_proto = 0.
    global_top5accs_proto = [0.] * num_parts

    global_top1acc_color = 0.
    global_top1accs_color = [0.] * num_parts
    global_top5acc_color = 0.
    global_top5accs_color = [0.] * num_parts

    global_ious = [0.] * num_parts
    global_iops = [0.] * num_parts
    global_iots = [0.] * num_parts

    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                     total=len(test_loader),
                     desc=progress_prefix + ' %s' % epoch,
                     mininterval=5.,
                     ncols=0)

    # Iterate through the test set
    for i, (x, x_aug, m, y) in test_iter:
        x, x_aug, m, y = x.to(device), x_aug.to(device), m.to(device), y.to(device)
        with torch.no_grad():
            proto_features, proto_pooled, color_features, color_pooled, agg, out = net(x, x_aug, m,
                                                                                       use_classification_layer,
                                                                                       aggregate)

            for part_i in range(proto_pooled.shape[1]):
                # Part accuracy metrics
                (top1accs_proto, top5accs_proto) = topk_accuracy(proto_pooled[:, part_i], y, topk=[1, 5])
                global_top1accs_proto[part_i] += torch.sum(top1accs_proto).item()
                global_top5accs_proto[part_i] += torch.sum(top5accs_proto).item()

                (top1accs_color, top5accs_color) = topk_accuracy(color_pooled[:, part_i], y, topk=[1, 5])
                global_top1accs_color[part_i] += torch.sum(top1accs_color).item()
                global_top5accs_color[part_i] += torch.sum(top5accs_color).item()

                # Distilled part segmentation metrics
                batch_size = proto_features.shape[0]
                feature_map_size = proto_features.shape[-1]
                part_features_idx = torch.argmax(proto_pooled[:, part_i, :], dim=1)
                part_features = proto_features[torch.arange(0, batch_size), part_i, part_features_idx, :, :]
                part_m = m[:, part_i, :, :]
                part_m = resize(part_m, [feature_map_size, feature_map_size], antialias=True)
                predicted_mask = torch.round(part_features)
                target_mask = torch.round(part_m)

                union_mask = torch.maximum(predicted_mask, target_mask)
                intersection_mask = torch.minimum(predicted_mask, target_mask)

                iou = intersection_mask.sum(-1).sum(-1) / union_mask.sum(-1).sum(-1)
                iou = torch.nan_to_num(iou, nan=1.0, posinf=1.0)

                iop = intersection_mask.sum(-1).sum(-1) / predicted_mask.sum(-1).sum(-1)
                iop = torch.nan_to_num(iop, nan=1.0, posinf=1.0)

                iot = intersection_mask.sum(-1).sum(-1) / target_mask.sum(-1).sum(-1)
                iot = torch.nan_to_num(iot, nan=1.0, posinf=1.0)

                global_ious[part_i] += torch.sum(iou).item()
                global_iops[part_i] += torch.sum(iop).item()
                global_iots[part_i] += torch.sum(iot).item()

            # Accuracy metrics
            (top1accs, top5accs) = topk_accuracy(out, y, topk=[1, 5])
            global_top1acc += torch.sum(top1accs).item()
            global_top5acc += torch.sum(top5accs).item()

            (top1acc_proto, top5acc_proto) = topk_accuracy(torch.mean(proto_pooled, dim=1), y, topk=[1, 5])
            global_top1acc_proto += torch.sum(top1acc_proto).item()
            global_top5acc_proto += torch.sum(top5acc_proto).item()

            (top1acc_color, top5acc_color) = topk_accuracy(torch.mean(color_pooled, dim=1), y, topk=[1, 5])
            global_top1acc_color += torch.sum(top1acc_color).item()
            global_top5acc_color += torch.sum(top5acc_color).item()

        del out
        del proto_pooled
        del proto_features
        del color_pooled
        del color_features
        del m
        del y

    info[f'top1_accuracy{mode}/ensemble'] = global_top1acc / len(test_loader.dataset)

    info[f'top5_accuracy{mode}/ensemble'] = global_top5acc / len(test_loader.dataset)

    for i, x in enumerate(global_top1accs_proto):
        info[f'top1_accuracy_proto{mode}/{i}'] = global_top1accs_proto[i] / len(test_loader.dataset)
    info[f'top1_accuracy_proto{mode}/ensemble'] = global_top1acc_proto / len(test_loader.dataset)

    for i, x in enumerate(global_top5accs_proto):
        info[f'top5_accuracy_proto{mode}/{i}'] = global_top5accs_proto[i] / len(test_loader.dataset)
    info[f'top5_accuracy_proto{mode}/ensemble'] = global_top5acc_proto / len(test_loader.dataset)

    for i, x in enumerate(global_top1accs_color):
        info[f'top1_accuracy_color{mode}/{i}'] = global_top1accs_color[i] / len(test_loader.dataset)
    info[f'top1_accuracy_color{mode}/ensemble'] = global_top1acc_color / len(test_loader.dataset)

    for i, x in enumerate(global_top5accs_color):
        info[f'top5_accuracy_color{mode}/{i}'] = global_top5accs_color[i] / len(test_loader.dataset)
    info[f'top5_accuracy_color{mode}/ensemble'] = global_top5acc_color / len(test_loader.dataset)

    for i, x in enumerate(global_ious):
        info[f'iou{mode}/{i}'] = global_ious[i] / len(test_loader.dataset)
    info[f'iou{mode}/mean'] = sum(global_ious) / num_parts / len(test_loader.dataset)

    for i, x in enumerate(global_iops):
        info[f'iop{mode}/{i}'] = global_iops[i] / len(test_loader.dataset)
    info[f'iop{mode}/mean'] = sum(global_iops) / num_parts / len(test_loader.dataset)

    for i, x in enumerate(global_ious):
        info[f'iot{mode}/{i}'] = global_iots[i] / len(test_loader.dataset)
    info[f'iot{mode}/mean'] = sum(global_iots) / num_parts / len(test_loader.dataset)

    return info
