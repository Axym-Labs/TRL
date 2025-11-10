import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from pytorch_lightning.loggers import WandbLogger

class MNISTModelReLU(pl.LightningModule):
    def __init__(self, batchnorm, lr, v3=False):
        super().__init__()
        self.lr = lr
        self.v3 = v3
        if not v3:
            self.init_v1(batchnorm)
        else:
            self.init_v3(batchnorm)

    def init_v1(self, batchnorm):
        self.flatten = nn.Flatten()
        h1 = 512
        self.fc1 = nn.Linear(28 * 28, h1)
        self.bn1 = nn.BatchNorm1d(h1) if batchnorm else nn.Identity()
        self.fc2 = nn.Linear(h1, 256)
        self.bn2 = nn.BatchNorm1d(256) if batchnorm else nn.Identity()
        self.fc3 = nn.Linear(256, 10)
    
    def init_v3(self, batchnorm):
        self.flatten = nn.Flatten()
        h1 = 2000
        h2 = 2000
        h3 = 2000
        h4 = 2000
        self.fc1 = nn.Linear(28 * 28, h1)
        self.bn1 = nn.BatchNorm1d(h1) if batchnorm else nn.Identity()
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2) if batchnorm else nn.Identity()
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3) if batchnorm else nn.Identity()
        self.fc4 = nn.Linear(h3, h4)
        self.bn4 = nn.BatchNorm1d(h4) if batchnorm else nn.Identity()
        self.fc5 = nn.Linear(h4, 10)

    def forward(self, x):
        if self.v3:
            return self.forward_v3(x)
        else: 
            return self.forward_v1(x)

    def forward_v1(self, x):
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def forward_v3(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.fc3(x)
        x = F.relu(self.bn3(x))
        x = self.fc4(x)
        x = F.relu(self.bn4(x))
        x = self.fc5(x)
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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

def run(epochs=60, batch_size=64, batchnorm=True, lr=15e-4, v3=True, seed=43):
    torch.manual_seed(seed)
    pl.seed_everything(seed)

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
    model_relu = MNISTModelReLU(batchnorm=batchnorm, lr=lr, v3=v3)
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