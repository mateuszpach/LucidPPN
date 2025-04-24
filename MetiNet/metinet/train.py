import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torchvision.transforms.functional import resize
from tqdm import tqdm


def train_metinet(net, train_loader, optimizer_net, optimizer_classifier, optimizer_color, scheduler_net,
                  scheduler_classifier, scheduler_color, no_color, part_criterion, class_criterion, epoch, device,
                  use_classification_layer, part_weight, proto_class_weight, color_class_weight, num_classes, num_parts,
                  aggregate, progress_prefix: str = 'Train Epoch'):
    # Make sure the model is in train mode
    net.train()

    # Store info about the procedure
    train_info = dict()
    total_acc = 0.
    total_proto_acc = 0.
    total_proto_accs = [0.] * num_parts
    total_color_acc = 0.
    total_color_accs = [0.] * num_parts
    total_loss = 0.
    total_part_loss = 0.
    total_part_losses = [0.] * num_parts
    total_proto_class_loss = 0.
    total_proto_class_losses = [0.] * num_parts
    total_color_class_loss = 0.
    total_color_class_losses = [0.] * num_parts
    lr_net = 0.
    lr_class = 0.
    lr_color = 0.

    # Show progress on progress bar
    iters = len(train_loader)
    train_iter = tqdm(enumerate(train_loader),
                      total=len(train_loader),
                      desc=progress_prefix + '%s' % epoch,
                      mininterval=2.,
                      ncols=0)

    # We call scheduler before epoch to avoid problems when starting from intermediate checkpoint
    scheduler_net.step()
    scheduler_color.step()

    # Loop over batches
    for i, (x, x_aug, m, y) in train_iter:
        x, x_aug, m, y = x.to(device), x_aug.to(device), m.to(device), y.to(device)

        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
        optimizer_color.zero_grad(set_to_none=True)

        # Perform a forward pass through the network
        proto_features, proto_pooled, color_features, color_pooled, agg, out = net(x, x_aug, m,
                                                                                   use_classification_layer, aggregate)
        results = calculate_loss(proto_features, proto_pooled, color_features, color_pooled, agg, out, m, y,
                                 num_classes, part_weight, proto_class_weight, color_class_weight, no_color,
                                 part_criterion, class_criterion, train_iter, use_classification_layer, epoch)
        (loss, part_loss, proto_class_loss, color_class_loss, part_losses, proto_class_losses, color_class_losses,
         acc, proto_acc, color_acc, proto_accs, color_accs) = results

        # Compute the gradient
        loss.backward()

        optimizer_net.step()
        lr_net = scheduler_net.get_last_lr()[0]

        optimizer_color.step()
        lr_color = scheduler_color.get_last_lr()[0]

        if use_classification_layer:
            optimizer_classifier.step()
            scheduler_classifier.step(epoch - 1 + (i / iters))
            lr_class = scheduler_classifier.get_last_lr()[0]

        # Collect stats
        with torch.no_grad():
            total_acc += acc
            total_proto_acc += proto_acc
            total_proto_accs = [x + y for x, y in zip(proto_accs, total_proto_accs)]
            total_color_acc += color_acc
            total_color_accs = [x + y for x, y in zip(color_accs, total_color_accs)]
            total_loss += loss.item()
            total_part_loss += part_loss
            total_part_losses = [x + y for x, y in zip(part_losses, total_part_losses)]
            total_proto_class_loss += proto_class_loss
            total_proto_class_losses = [x + y for x, y in zip(proto_class_losses, total_proto_class_losses)]
            total_color_class_loss += color_class_loss
            total_color_class_losses = [x + y for x, y in zip(color_class_losses, total_color_class_losses)]


    # Report stats
    train_info['train_accuracy/mean'] = total_acc / iters
    train_info['train_proto_accuracy/mean'] = total_proto_acc / iters
    for i, x in enumerate(total_proto_accs):
        train_info[f'train_proto_accuracy/{i}'] = total_proto_accs[i] / iters
    train_info['train_color_accuracy/mean'] = total_color_acc / iters
    for i, x in enumerate(total_color_accs):
        train_info[f'train_color_accuracy/{i}'] = total_color_accs[i] / iters
    train_info['loss/mean'] = total_loss / iters
    train_info['part_loss/mean'] = total_part_loss / iters
    train_info['proto_class_loss/mean'] = total_proto_class_loss / iters
    train_info['color_class_loss/mean'] = total_color_class_loss / iters
    for i, x in enumerate(total_part_losses):
        train_info[f'part_loss/{i}'] = total_part_losses[i] / iters
    for i, x in enumerate(total_proto_class_losses):
        train_info[f'proto_class_loss/{i}'] = total_proto_class_losses[i] / iters
    for i, x in enumerate(total_color_class_losses):
        train_info[f'color_class_loss/{i}'] = total_color_class_losses[i] / iters
    train_info['lrs_net'] = lr_net
    train_info['lrs_class'] = lr_class
    train_info['lrs_color'] = lr_color

    return train_info


def calculate_loss(proto_features, proto_pooled, color_features, color_pooled, agg, out, m, y, num_classes,
                   part_weight, proto_class_weight, color_class_weight, no_color,
                   part_criterion, class_criterion, train_iter, use_classification_layer, epoch):
    y_onehot = F.one_hot(y, num_classes=num_classes).to(dtype=torch.float32)
    batch_size = proto_features.shape[0]

    # Part alignment loss
    feature_map_size = proto_features.shape[-1]
    part_losses = []
    if part_weight > 0:
        for part_i in range(proto_features.shape[1]):
            part_proto_features = proto_features[torch.arange(0, batch_size), part_i, y, :, :]
            target = m[:, part_i, :, :]
            target = resize(target, [feature_map_size, feature_map_size], antialias=True)
            target = torch.clamp(target, min=0, max=1)
            part_losses.append(part_criterion(part_proto_features, target))
        part_loss = part_weight * sum(part_losses) / len(part_losses)
    else:
        for part_i in range(proto_features.shape[1]):
            part_losses.append(torch.tensor(0).to(device=proto_features.device))
        part_loss = part_weight * sum(part_losses) / len(part_losses)

    # Prototype classification loss
    proto_class_losses = []
    for part_i in range(proto_pooled.shape[1]):
        part_proto_pooled = proto_pooled[:, part_i, :]
        proto_class_losses.append(class_criterion(part_proto_pooled, y_onehot))
    proto_class_loss = proto_class_weight * sum(proto_class_losses) / len(proto_class_losses)

    # Color classification loss
    color_class_losses = []
    for part_i in range(color_features.shape[1]):
        part_color_pooled = color_pooled[:, part_i, :]
        color_class_losses.append(class_criterion(part_color_pooled, y_onehot))
    color_class_loss = color_class_weight * sum(color_class_losses) / len(color_class_losses)

    # TODO: Classification loss for linear layer
    # if use_classification_layer:
    #     class_losses = [class_criterion(pooled[:, part_i],
    #                                     F.one_hot(y, num_classes=num_classes).to(dtype=torch.float32))
    #                     for part_i in range(pooled.shape[1])]
    #     class_loss = class_weight * sum(class_losses)
    # print(epoch)
    # if epoch > 6:
    #     loss = color_class_loss
    # else:
    #     loss = proto_class_loss + color_class_loss

    loss = part_loss + proto_class_loss + color_class_loss

    # Accuracy stats
    acc = torch.sum(torch.eq(torch.argmax(out, dim=1), y)).item() / float(len(y))

    # Prototype accuracy stats
    proto_accs = []
    for part_i in range(proto_pooled.shape[1]):
        torch.sum(torch.eq(torch.argmax(proto_pooled[:, part_i], dim=1), y)).item() / float(len(y))
    proto_acc = torch.sum(torch.eq(torch.argmax(torch.mean(proto_pooled, dim=1), dim=1), y)).item() / float(len(y))

    # Color accuracy stats
    color_accs = []
    for part_i in range(color_pooled.shape[1]):
        torch.sum(torch.eq(torch.argmax(color_pooled[:, part_i], dim=1), y)).item() / float(len(y))
    color_acc = torch.sum(torch.eq(torch.argmax(torch.mean(color_pooled, dim=1), dim=1), y)).item() / float(len(y))

    train_iter.set_postfix_str(f'L:{loss.item():.3f}, '
                               f'PL:{part_loss.item():.3f}, '
                               f'PCL:{proto_class_loss.item():.3f}, '
                               f'CCL:{color_class_loss.item():.3f}, '
                               f'A:{acc:.3f}', refresh=False)

    return (loss, part_loss, proto_class_loss, color_class_loss, part_losses, proto_class_losses, color_class_losses,
            acc, proto_acc, color_acc, proto_accs, color_accs)
