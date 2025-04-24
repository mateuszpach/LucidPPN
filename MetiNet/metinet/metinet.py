import argparse
import torch.nn as nn
import torch.nn.functional as F
from features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, \
    resnet101_features, resnet152_features
from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features
import torch
from torch import Tensor
from torchvision.transforms.functional import resize


# from https://github.com/limacv/RGB_HSV_HSL/blob/master/color_torch.py
def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def rgb_to_hue(rgb: torch.Tensor) -> torch.Tensor:
    hsv = rgb2hsv_torch(rgb)
    hue = hsv[:, 0:1, :, :]
    return hue


class MetiNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_parts: int,
                 feature_net: nn.Module,
                 add_on_layers: nn.Module,
                 pool_layer: nn.Module,
                 classification_layer: nn.Module,
                 color_net: nn.Module):
        super().__init__()
        assert num_classes > 0
        self._num_classes = num_classes
        self._num_parts = num_parts
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer
        self._color_net = color_net

    def forward(self, x, x_aug, m, use_classification_layer=False, aggregate='mean'):
        bs = x_aug.shape[0]

        # 1. Prototype features and scores
        features = self._net(x_aug)
        # bs x num_parts * num_classes x h x w
        proto_features = self._add_on(features)
        # bs x num_parts x num_classes x h x w
        grouped_proto_features = proto_features.unflatten(1, (self._num_parts, self._num_classes))
        # bs x num_parts * num_classes
        proto_pooled = self._pool(proto_features)
        # bs x num_parts x num_classes
        grouped_proto_pooled = proto_pooled.unflatten(1, (self._num_parts, self._num_classes))

        if x is not None:
            h, w = grouped_proto_features.shape[3:5]

            # bs x 3 x h x w
            x_resized = resize(x, [h, w], antialias=True)

            # only hue for colornet
            # x_resized = rgb_to_hue(x_resized).repeat(1, 3, 1, 1)

            # bs * h * w x 3
            color_net_input = x_resized.permute(0, 2, 3, 1).flatten(0, 2)
            # bs * h * w x num_parts * num_classes
            color_net_output = self._color_net(color_net_input)
            # bs x num_parts * num_classes x h x w
            color_features = color_net_output.unflatten(0, (bs, h, w)).permute(0, 3, 1, 2)

            # bs x num_parts x 1 x h x w
            # mean_grouped_proto_features = torch.max(grouped_proto_features, dim=2, keepdim=True)[0]
            # print(mean_grouped_proto_features.shape)
            # print(grouped_proto_features.shape)
            # bs x num_parts x num_classes x h x w
            # mean_grouped_proto_features = mean_grouped_proto_features.repeat(*grouped_proto_features.shape)
            # bs x num_parts * num_classes x h x w
            # mean_proto_features = mean_grouped_proto_features.flatten(1, 2)

            # color_features = ((color_features.unflatten(1, (self._num_parts, self._num_classes)) * 2) * (mean_grouped_proto_features / 2)).flatten(1, 2)

            # bs x num_parts * num_classes x h x w
            # r1, r2 = 0.0, 1.0
            # color_features = color_features * (proto_features + ((r1 - r2) * torch.rand(bs, self._num_parts * self._num_classes, 1, 1).to(color_features.device) + r2)) / 2
            color_features = color_features * proto_features.detach()


            # uncomment to use pdisconet masks instead
            # mask_resized = resize(m, [h, w], antialias=True)
            # mask_resized = torch.clamp(mask_resized, min=0, max=1)
            # mask_resized = mask_resized.unsqueeze(2).repeat(1, 1, self._num_classes, 1, 1).flatten(1, 2)
            # color_features = color_features * mask_resized

            # bs x num_parts x num_classes x h x w
            grouped_color_features = color_features.unflatten(1, (self._num_parts, self._num_classes))
            # bs x num_parts * num_classes
            color_pooled = self._pool(color_features)
            # bs x num_parts x num_classes
            grouped_color_pooled = color_pooled.unflatten(1, (self._num_parts, self._num_classes))
        else:
            grouped_color_features = grouped_proto_features
            grouped_color_pooled = grouped_proto_pooled

        # 4. Aggregated scores
        if aggregate == 'mean':
            # agg = (grouped_color_pooled + grouped_proto_pooled) / 2
            agg = grouped_color_pooled
        elif aggregate == 'product':
            # agg = grouped_color_pooled * grouped_proto_pooled
            agg = grouped_color_pooled
        else:
            raise Exception("Available options for aggregate are: 'mean', 'product'")

        if use_classification_layer:
            out = self._classification(agg)
        else:
            out = torch.mean(agg, dim=1)

        return grouped_proto_features, grouped_proto_pooled, grouped_color_features, grouped_color_pooled, agg, out


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'convnext_tiny_26': convnext_tiny_26_features,
                                 'convnext_tiny_13': convnext_tiny_13_features}


# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.relu(self.weight), self.bias)


def get_network(num_classes: int, args: argparse.Namespace):
    features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)
    features_name = str(features).upper()
    if 'next' in args.net:
        features_name = str(args.net).upper()
    if features_name.startswith('RES') or features_name.startswith('CONVNEXT'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    else:
        raise Exception('other base architecture NOT implemented')

    num_prototypes = args.num_classes * args.num_parts
    add_on_layers = nn.Sequential(
        nn.Conv2d(in_channels=first_add_on_layer_in_channels,
                  out_channels=num_prototypes,
                  kernel_size=1, stride=1, padding=0, bias=True),
        nn.Sigmoid(),
    )
    pool_layer = nn.Sequential(
        nn.AdaptiveMaxPool2d(output_size=(1, 1)),
        nn.Flatten()
    )

    if args.bias:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=True)
    else:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)

    # original
    color_net = nn.Sequential(
        nn.Linear(3, 20),
        nn.ReLU(),
        nn.Linear(20, 50),
        nn.ReLU(),
        nn.Linear(50, 150),
        nn.ReLU(),
        nn.Linear(150, 200),
        nn.ReLU(),
        nn.Linear(200, 600),
        nn.ReLU(),
        nn.Linear(600, args.num_parts * args.num_classes),
        nn.Sigmoid()
    )

    # linear
    # color_net = nn.Sequential(
    #     nn.Linear(3, args.num_parts * args.num_classes),
    #     nn.Sigmoid()
    # )

    # # 2 layers
    # color_net = nn.Sequential(
    #     nn.Linear(3, 600),
    #     nn.ReLU(),
    #     nn.Linear(600, args.num_parts * args.num_classes),
    #     nn.Sigmoid()
    # )
    #
    # # 4 layers
    # color_net = nn.Sequential(
    #     nn.Linear(3, 50),
    #     nn.ReLU(),
    #     nn.Linear(50, 200),
    #     nn.ReLU(),
    #     nn.Linear(200, 600),
    #     nn.ReLU(),
    #     nn.Linear(600, args.num_parts * args.num_classes),
    #     nn.Sigmoid()
    # )

    return features, add_on_layers, pool_layer, classification_layer, color_net, num_prototypes
