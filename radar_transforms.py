import torchvision

from torchvision import transforms, utils


def radar_transform(max_prec_val):
    transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t ** (1 / 3)),
            transforms.Lambda(lambda t: t / max_prec_val),  # Scale between [0, 1]
            transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
        ]
    )

    return transform


def reverse_transform(max_prec_val):
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t * max_prec_val),
            transforms.Lambda(lambda t: t ** 3),
        ]
    )
    return reverse_transforms


def conditional_embedding_transform(max_prec_val):
    transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)),
            transforms.Lambda(lambda t: t ** (1 / 3)),
            transforms.Lambda(lambda t: t / max_prec_val),  # Scale between [0, 1]
            transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
        ]
    )

    return transform

