import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from pytorch_lightning.loggers import WandbLogger

class MNISTModelReLU(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.init_deep()

    def init_old(self):
        self.flatten = nn.Flatten()
        # to match vicreg_10_nobn parameter count roughly
        # or 512 to match its hidden dim
        h1 = 824
        self.fc1 = nn.Linear(28 * 28, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)

    def init_deep(self):
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        # self.fc5 = nn.Linear(128, 128)
        # self.bn5 = nn.BatchNorm1d(128)
        # self.fc6 = nn.Linear(128, 128)
        # self.bn6 = nn.BatchNorm1d(128)
        # self.fc7 = nn.Linear(128, 128)
        # self.bn7 = nn.BatchNorm1d(128)
        # self.fc8 = nn.Linear(128, 128)
        # self.bn8 = nn.BatchNorm1d(128)
        # self.fc9 = nn.Linear(128, 128)
        # self.bn9 = nn.BatchNorm1d(128)
        self.fc10 = nn.Linear(128, 10)

    def forward(self, x):
        return self.forward_deep(x)

    def forward_shallow(self, x):
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def forward_deep(self, x):
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        # x = F.relu(self.bn5(self.fc5(x)))
        # x = F.relu(self.bn6(self.fc6(x)))
        # x = F.relu(self.bn7(self.fc7(x)))
        # x = F.relu(self.bn8(self.fc8(x)))
        # x = F.relu(self.bn9(self.fc9(x)))
        x = self.fc10(x)
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

def main():
    # Load MNIST dataset
    # no augmentation: it decreases validation performance
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=get_mnist_val_transform())
    val_dataset = datasets.MNIST('data', train=False, transform=get_mnist_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project="experiments_mnist")

    # Train ReLU model
    print("Training ReLU Model...")
    model_relu = MNISTModelReLU()
    trainer_relu = pl.Trainer(
        max_epochs=40,
        logger=wandb_logger,
        accelerator='auto',
    )
    trainer_relu.fit(model_relu, train_loader, val_loader)

    # Start a new wandb run for the Linear model
    wandb.finish()
    wandb_logger = WandbLogger(project="experiments_mnist")

if __name__ == '__main__':
    main()