from torchvision import transforms
import torch


class Dropout:
    def __init__(self, dropout_prob, p):
        self.dropout_prob = dropout_prob
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            mask = ~torch.bernoulli(torch.full_like(img, self.dropout_prob)).bool()
            return img * mask
        else:
            return img


class NviewsAugment(object):
    def __init__(self, transforms, n_views=2):
        self.transforms = transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.transforms(x) for i in range(self.n_views)]


contrastive_base = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=256, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=11),  # , sigma=(0.1, 1.)),
        transforms.ToTensor(),
        Dropout(0.1, 0.3),
    ]
)


contrastive_wsi_transforms = NviewsAugment(contrastive_base, n_views=2)
