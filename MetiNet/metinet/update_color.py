import torch
import torch.optim
import torch.utils.data
from tqdm import tqdm


def update_color_metinet(net, train_loader, epoch, device,
                         progress_prefix: str = 'Update Color Epoch'):
    # Make sure the model is in eval mode
    net.eval()

    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader),
                      total=len(train_loader),
                      desc=progress_prefix + '%s' % epoch,
                      mininterval=2.,
                      ncols=0)

    color_cubes = torch.zeros_like(net.module._color_cubes, device=device)

    # Loop over batches
    k = 6
    for i, (x, x_aug, m, y) in train_iter:
        x, x_aug, m, y = x.to(device), x_aug.to(device), m.to(device), y.to(device)
        with torch.no_grad():
            proto_features, pooled, mean_rgb, color_features, out = net(x, x_aug)

            # batch_size x num_parts x 3
            mean_rgb = torch.floor(mean_rgb * k).to(dtype=torch.int32)

            for sample_i in range(mean_rgb.shape[0]):
                for part_i in range(mean_rgb.shape[1]):
                    sample_mean_rgb = mean_rgb[sample_i, part_i]
                    sample_r, sample_g, sample_b = sample_mean_rgb[0], sample_mean_rgb[1], sample_mean_rgb[2]
                    color_cubes[part_i, y[sample_i], sample_r, sample_g, sample_b] = 1

    net.module._color_cubes = color_cubes

