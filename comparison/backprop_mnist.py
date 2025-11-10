import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from pytorch_lightning.loggers import WandbLogger

class MNISTModelReLU(pl.LightningModule):
    def __init__(self, batchnorm, lr):
        super().__init__()
        self.lr = lr

        self.flatten = nn.Flatten()
        h1 = 512
        self.fc1 = nn.Linear(28 * 28, h1)
        self.bn1 = nn.BatchNorm1d(h1) if batchnorm else nn.Identity()
        self.fc2 = nn.Linear(h1, 256)
        self.bn2 = nn.BatchNorm1d(256) if batchnorm else nn.Identity()
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)

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

def run(epochs=20, batch_size=64, batchnorm=True, lr=1e-3):
    # Load MNIST dataset
    # no augmentation: it decreases validation performance
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=get_mnist_val_transform())
    val_dataset = datasets.MNIST('data', train=False, transform=get_mnist_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project="experiments_mnist")

    # Train ReLU model
    print("Training ReLU Model...")
    model_relu = MNISTModelReLU(batchnorm=batchnorm, lr=lr)
    trainer_relu = pl.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        accelerator='auto',
        check_val_every_n_epoch=epochs
    )
    trainer_relu.fit(model_relu, train_loader, val_loader)

    wandb.finish()

    return trainer_relu.callback_metrics.get('val_acc').item()



if __name__ == '__main__':
    run()