import torch
import torchvision.transforms as transforms

def get_patch_size(args):
    patchsize = 32
    skip = round((args.image_size - patchsize) / (args.wshape-1))
    return patchsize, skip

def init_weights_xavier(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

# https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b?permalink_comment_id=3662215#gistcomment-3662215
def topk_accuracy(output, target, topk=[1,]):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        topk2 = [x for x in topk if x <= output.shape[1]] #ensures that k is not larger than number of classes
        maxk = max(topk2)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

        res = []
        for k in topk:
            if k in topk2:
                correct_k = correct[:k].reshape(-1).float()
                res.append(correct_k)
            else:
                res.append(torch.zeros_like(target))
        return res

def random_hue_jitter(image_orig, max_hue_shift=0.5):
    """
    Apply random hue jittering without changing the luminance.

    Args:
        image (PIL Image): Input image.
        max_hue_shift (float): Maximum amount of hue shift. Should be in [0, 0.5].

    Returns:
        PIL Image: Hue jittered image.
    """

    gray_transformed = transforms.ToTensor()(transforms.Grayscale(3)(image_orig))

    image = transforms.ToTensor()(image_orig)

    hue_transformed = transforms.ColorJitter(hue=max_hue_shift)(image)
    gray_hue_transformed = transforms.Grayscale(3)(hue_transformed)

    image = hue_transformed
    image = image * gray_transformed / gray_hue_transformed

    image = torch.nan_to_num(image, nan=0.0)

    return image
