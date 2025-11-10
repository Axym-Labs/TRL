import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger

# --------------------------------------------------------------------------
# 1. Data Augmentation & Dataset Wrapper
# --------------------------------------------------------------------------

class TwoCropsTransform:
    """Take two random augmented views of one image."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        # Apply the same series of random transformations twice to the same input image
        x1 = self.base_transform(x)
        x2 = self.base_transform(x)
        return [x1, x2]

def get_mnist_transforms():
    """Get the augmentation pipeline for VICReg pre-training on MNIST."""
    # Simplified augmentations suitable for MNIST.
    # We avoid horizontal flips which can change the digit's meaning (e.g., 6 vs. 9).
    return transforms.Compose([
        transforms.RandomResizedCrop(28, scale=(0.5, 1.0)),
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST normalization
    ])

# --------------------------------------------------------------------------
# 2. Loss Function
# --------------------------------------------------------------------------

def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class VICRegLoss(nn.Module):
    """The VICReg loss function."""
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z_a, z_b):
        batch_size, num_features = z_a.shape

        # 1. Invariance Loss (Mean Squared Error)
        sim_loss = F.mse_loss(z_a, z_b)

        # 2. Variance Loss (Hinge Loss)
        std_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
        std_b = torch.sqrt(z_b.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_a)) / 2 + torch.mean(F.relu(1 - std_b)) / 2

        # 3. Covariance Loss
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_a = (z_a.T @ z_a) / (batch_size - 1)
        cov_b = (z_b.T @ z_b) / (batch_size - 1)
        cov_loss = (off_diagonal(cov_a).pow_(2).sum() / num_features +
                    off_diagonal(cov_b).pow_(2).sum() / num_features)

        # Weighted sum of the three losses
        loss = (self.sim_coeff * sim_loss +
                self.std_coeff * std_loss +
                self.cov_coeff * cov_loss)
        
        return loss, sim_loss, std_loss, cov_loss

# --------------------------------------------------------------------------
# 3. Architecture (Encoder & Expander)
# --------------------------------------------------------------------------

class SmallCNNEncoder(nn.Module):
    """A small CNN encoder for MNIST."""
    def __init__(self, rep_dim=128):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32), nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64), nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.BatchNorm2d(64), nn.ReLU(),
            # nn.Flatten()
        )

    def forward(self, x):
        return self.convnet(x)

class Expander(nn.Module):
    """The expander network projects representations to a higher-dimensional space."""
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------------------------------
# 4. Lightning Module with Online Linear Evaluation
# --------------------------------------------------------------------------

class VICRegMNIST(pl.LightningModule):
    def __init__(self, rep_dim=256, emb_dim=512, lr=1e-3, batch_size=256, num_classes=10):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = SmallCNNEncoder(rep_dim=rep_dim)
        self.expander = Expander(input_dim=rep_dim, output_dim=emb_dim)
        self.criterion = VICRegLoss()
        
        # Linear classifier for online evaluation
        self.linear_evaluator = nn.Linear(rep_dim, num_classes)
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        """Returns the representation for downstream tasks."""
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        (x1, x2), y = batch

        x1 = x1.view(x1.size(0), -1) # Flatten the images for the linear encoder
        x2 = x2.view(x2.size(0), -1)
        # --- VICReg Loss Calculation (updates encoder + expander) ---
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)
        z1 = self.expander(y1)
        z2 = self.expander(y2)
        vicreg_loss, sim, std, cov = self.criterion(z1, z2)

        # --- Online Linear Evaluation (updates linear_evaluator only) ---
        # The key is to stop gradients from the classifier flowing back to the encoder.
        with torch.no_grad():
            y1_frozen = self.encoder(x1)
        
        logits = self.linear_evaluator(y1_frozen)
        ce_loss = self.classification_loss(logits, y)

        # --- Combine Losses ---
        # Autograd will correctly route gradients:
        # vicreg_loss -> encoder, expander
        # ce_loss -> linear_evaluator
        total_loss = vicreg_loss + ce_loss
        
        # --- Logging ---
        self.log('vicreg_loss', vicreg_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('classification_loss', ce_loss, on_step=True, on_epoch=False)
        self.log('sim_loss', sim, on_step=False, on_epoch=True)
        self.log('std_loss', std, on_step=False, on_epoch=True)
        self.log('cov_loss', cov, on_step=False, on_epoch=True)
        
        # Log training accuracy
        preds = torch.argmax(logits, dim=1)
        train_acc = (preds == y).float().mean()
        self.log('train_acc', train_acc, prog_bar=True, on_step=True, on_epoch=False)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)  # Flatten the images for the linear encoder
        y_rep = self.encoder(x) # Gradients are off by default in validation
        logits = self.linear_evaluator(y_rep)
        
        preds = torch.argmax(logits, dim=1)
        val_acc = (preds == y).float().mean()
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # A single optimizer can train both parts of the network
        # because the gradient flow is controlled in the training_step.
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        return optimizer

# --------------------------------------------------------------------------
# 5. Execution
# --------------------------------------------------------------------------

if __name__ == '__main__':
    pl.seed_everything(42)
    BATCH_SIZE = 64
    MAX_EPOCHS = 60
    LR = 15e-4
    DATA_PATH = './data'
    os.makedirs(DATA_PATH, exist_ok=True)

    # --- Prepare Data ---
    # Training data with special two-crop augmentation for VICReg
    train_transform = get_mnist_transforms()
    train_dataset = torchvision.datasets.MNIST(
        DATA_PATH, train=True, download=True, transform=TwoCropsTransform(train_transform)
    )
    
    # Validation/Test data with simple standard transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    val_dataset = torchvision.datasets.MNIST(
        DATA_PATH, train=False, download=True, transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # --- Initialize Model & Trainer ---
    model = VICRegMNIST(batch_size=BATCH_SIZE, lr=LR)
    wandb_logger = WandbLogger(project="experiments_mnist")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        logger=wandb_logger, # Enable logger to see epoch-level metrics
        check_val_every_n_epoch=MAX_EPOCHS
    )

    print("Starting VICReg pre-training on MNIST with online linear evaluation...")
    trainer.fit(model, train_loader, val_loader)
    print("Pre-training finished.")
    
    # --- Final Validation Accuracy ---
    final_val_acc = trainer.callback_metrics.get('val_acc')
    if final_val_acc:
        print(f"\nFinal validation accuracy of the linear probe: {final_val_acc.item():.4f}")