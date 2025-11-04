from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from trl.config.config import DataConfig
from trl.data import CoherentSampler

def get_mnist_augment():
    return transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_mnist_val_transform():
    return transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])

def build_dataloaders(cfg: DataConfig):
    assert cfg.batch_size % cfg.chunk_size == 0, "batch_size must be a multiple of chunk_size"

    # For pretraining we need a base dataset without transforms so the wrapper can sample two imgs
    mnist_dataset_transform_specific = lambda t: torchvision.datasets.MNIST(cfg.data_path, train=True, download=True, transform=t)
    base_train = mnist_dataset_transform_specific(get_mnist_augment())
    head_base_train = mnist_dataset_transform_specific(get_mnist_val_transform())

    # extract labels for sampler
    label_list = [base_train[i][1] for i in range(len(base_train))]

    # sampler that enforces chunk coherence
    sampler = CoherentSampler(label_list=label_list, coherence=cfg.coherence, chunk_size=cfg.chunk_size, shuffle=True)

    val_ds = torchvision.datasets.MNIST(cfg.data_path, train=False, download=True, transform=get_mnist_val_transform())

    train_loader_kwargs = dict(
        batch_size=cfg.batch_size, 
        sampler=sampler,
        num_workers=cfg.num_workers, 
        pin_memory=cfg.pin_memory, 
        drop_last=True,
    )
    train_loader = DataLoader(base_train, **train_loader_kwargs)
    head_train_loader = DataLoader(head_base_train, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    return train_loader, head_train_loader, val_loader

