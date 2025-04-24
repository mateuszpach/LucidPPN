import torch
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.func import topk_accuracy
from torchvision.transforms.functional import resize
import wandb

@torch.no_grad()
def final_eval_metinet(net,
                       test_loader: DataLoader,
                       epoch,
                       device,
                       use_classification_layer,
                       num_classes,
                       aggregate,
                       mode,
                       progress_prefix: str = 'Eval Epoch') -> dict:
    net = net.to(device)

    # Make sure the model is in evaluation mode
    net.eval()

    # Keep an info dict about the procedure
    info = dict()

    class_proto_accs = [0.] * num_classes
    class_color_accs = [0.] * num_classes
    class_ensemble_accs = [0.] * num_classes

    class_n = [0.] * num_classes

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

            for pp, cp, en, yy in zip(torch.mean(proto_pooled, dim=1),
                                      torch.mean(color_pooled, dim=1),
                                      out, y):
                label = int(yy.item())
                class_n[label] += 1
                class_proto_accs[label] += topk_accuracy(pp.unsqueeze(0), yy.unsqueeze(0), topk=[1])[0].item()
                class_color_accs[label] += topk_accuracy(cp.unsqueeze(0), yy.unsqueeze(0), topk=[1])[0].item()
                class_ensemble_accs[label] += topk_accuracy(en.unsqueeze(0), yy.unsqueeze(0), topk=[1])[0].item()

        del out
        del proto_pooled
        del proto_features
        del color_pooled
        del color_features
        del m
        del y

    for i in range(len(class_proto_accs)):
        class_proto_accs[i] = class_proto_accs[i] / class_n[i]
        class_color_accs[i] = class_color_accs[i] / class_n[i]
        class_ensemble_accs[i] = class_ensemble_accs[i] / class_n[i]

    improved = 0
    worsened = 0
    data = []
    for i in range(len(class_proto_accs)):
        data.append([i, class_proto_accs[i], class_color_accs[i], class_ensemble_accs[i]])
        if class_color_accs[i] > class_proto_accs[i]:
            improved += 1
        if class_color_accs[i] < class_proto_accs[i]:
            worsened += 1

    classes_table = wandb.Table(columns=['class', 'proto', 'color', 'ensemble'], data=data)

    info[f'class_accuracy/table'] = classes_table
    info[f'class_accuracy/proto'] = sum(class_proto_accs) / len(class_proto_accs)
    info[f'class_accuracy/color'] = sum(class_color_accs) / len(class_color_accs)
    info[f'class_accuracy/ensemble'] = sum(class_ensemble_accs) / len(class_ensemble_accs)
    info[f'class_accuracy/worsened'] = worsened
    info[f'class_accuracy/improved'] = improved

    return info
