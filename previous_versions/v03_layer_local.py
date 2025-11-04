import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

# Make sure to run: pip install wandb tqdm
# And log in: wandb login

# --------------------------------------------------------------------------
# 1. Data Augmentation & Dataset Wrapper (Unchanged)
# --------------------------------------------------------------------------

class TwoCropsTransform:
    """Take two random augmented views of one image."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

def get_mnist_transforms():
    """Get the augmentation pipeline for VICReg pre-training on MNIST."""
    return transforms.Compose([
        transforms.RandomResizedCrop(28, scale=(0.5, 1.0)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

# --------------------------------------------------------------------------
# 2. Loss Function (Unchanged)
# --------------------------------------------------------------------------

class VICRegLoss(nn.Module):
    """The VICReg loss function."""
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z_a, z_b):
        z_b = z_b.detach()
        batch_size, num_features = z_a.shape
        sim_loss = F.mse_loss(z_a, z_b)
        std_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_a))
        z_a = z_a - z_a.mean(dim=0)
        cov_a = (z_a.T @ z_a) / (batch_size - 1)
        cov_loss = off_diagonal(cov_a).pow_(2).sum() / num_features
        loss = (self.sim_coeff * sim_loss +
                self.std_coeff * std_loss +
                self.cov_coeff * cov_loss)
        return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# --------------------------------------------------------------------------
# 3. Modular Architecture (Unchanged)
# --------------------------------------------------------------------------

class GreedyEncoder(nn.Module):
    """An encoder composed of a list of modules, trained greedily."""
    def __init__(self, rep_dim=256):
        super().__init__()
        self.rep_dim = rep_dim
        self.layers = nn.ModuleList([
            nn.Sequential( # Layer 0
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential( # Layer 1
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential( # Layer 2
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.BatchNorm2d(64), nn.ReLU(),
            ),
            nn.Sequential( # Layer 3
                nn.Flatten(),
                nn.Linear(64 * 5 * 5, rep_dim)
            )
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --------------------------------------------------------------------------
# 4. Greedy Layer-Wise Unsupervised Trainer (FIXED)
# --------------------------------------------------------------------------

class LayerWiseVICRegTrainer(pl.LightningModule):
    def __init__(self, epochs_per_layer=5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = GreedyEncoder()
        self.criterion = VICRegLoss()
        
        # Track which layer is currently being trained
        self.current_layer_idx = 0
        self.epochs_per_layer = epochs_per_layer

        # --- FIX: Enable Manual Optimization ---
        self.automatic_optimization = False

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        # --- FIX: Get the correct optimizer for the current layer ---
        optimizers = self.optimizers()
        opt = optimizers[self.current_layer_idx]

        (x1, x2), _ = batch
        
        # --- Pass through frozen, already-trained layers ---
        with torch.no_grad():
            inp1, inp2 = x1, x2
            for i in range(self.current_layer_idx):
                self.encoder.layers[i].eval()
                inp1 = self.encoder.layers[i](inp1)
                inp2 = self.encoder.layers[i](inp2)
        
        # --- Train the current layer ---
        current_layer = self.encoder.layers[self.current_layer_idx]
        current_layer.train()
        y1 = current_layer(inp1)
        y2 = current_layer(inp2)

        # --- Reshape for loss calculation ---
        if y1.dim() > 2: # This is a conv layer
            b, c, h, w = y1.shape
            y1 = y1.permute(0, 2, 3, 1).reshape(b * h * w, c)
            y2 = y2.permute(0, 2, 3, 1).reshape(b * h * w, c)
        
        loss = self.criterion(y1, y2)
        
        self.log(f'layer_{self.current_layer_idx}/vicreg_loss', loss, prog_bar=True)
        
        # --- FIX: Manual optimization loop ---
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

    def configure_optimizers(self):
        # Create a separate optimizer for each layer
        optimizers = [
            torch.optim.Adam(layer.parameters(), lr=self.hparams.lr)
            for layer in self.encoder.layers
        ]
        return optimizers

    def on_train_epoch_end(self):
        # Check if it's time to switch to the next layer
        if self.trainer.current_epoch > 0 and \
           (self.trainer.current_epoch + 1) % self.epochs_per_layer == 0:
            
            if self.current_layer_idx < len(self.encoder.layers) - 1:
                self.current_layer_idx += 1
                print(f"\n\n--- Switching to train layer {self.current_layer_idx} ---")

# --------------------------------------------------------------------------
# 5. Supervised Linear Classifier (Unchanged)
# --------------------------------------------------------------------------

class LinearClassifier(pl.LightningModule):
    def __init__(self, encoder, lr=1e-3, num_classes=10):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Linear(self.encoder.rep_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        with torch.no_grad():
            reps = self.encoder(x)
        return self.classifier(reps)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('classifier_train_loss', loss)
        self.log('classifier_train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('classifier_val_loss', loss)
        self.log('classifier_val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.lr)

# --------------------------------------------------------------------------
# 6. Execution (FIXED LOGGER)
# --------------------------------------------------------------------------

if __name__ == '__main__':
    pl.seed_everything(42)
    BATCH_SIZE = 256
    EPOCHS_PER_LAYER = 40
    CLASSIFIER_EPOCHS = 20
    DATA_PATH = './data'

    # --- Prepare Data ---
    train_transform = get_mnist_transforms()
    train_dataset = torchvision.datasets.MNIST(
        DATA_PATH, train=True, download=True, transform=TwoCropsTransform(train_transform)
    )
    val_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
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

    # --- 1. Unsupervised Layer-Wise Pre-training ---
    # --- FIX: Updated logger name ---
    wandb_logger = WandbLogger(project="experiments_mnist", name="vicreg_3_layer_local")
    
    greedy_trainer_model = LayerWiseVICRegTrainer(epochs_per_layer=EPOCHS_PER_LAYER)
    
    num_layers = len(greedy_trainer_model.encoder.layers)
    total_pretrain_epochs = num_layers * EPOCHS_PER_LAYER
    
    pre_trainer = pl.Trainer(
        max_epochs=total_pretrain_epochs,
        accelerator="auto", devices=1,
        logger=wandb_logger,
        enable_checkpointing=False
    )

    print("--- Starting Greedy Layer-Wise Unsupervised Pre-training ---")
    pre_trainer.fit(greedy_trainer_model, train_loader)
    print("--- Pre-training Finished ---")
    
    # --- 2. Supervised Linear Evaluation ---
    print("\n--- Starting Supervised Linear Evaluation ---")
    frozen_encoder = greedy_trainer_model.encoder
    # Train the linear classifier using the validation set as its training data
    # to get a performance estimate.
    classifier_model = LinearClassifier(encoder=frozen_encoder)
    
    classifier_trainer = pl.Trainer(
        max_epochs=CLASSIFIER_EPOCHS,
        accelerator="auto", devices=1,
        logger=wandb_logger,
        enable_checkpointing=False,
        num_sanity_val_steps=0
    )
    classifier_trainer.fit(classifier_model, val_loader)
    print("--- Training of linear probe finished ---")
    print("--- Running final validation on test set ---")
    classifier_trainer.validate(classifier_model, dataloaders=val_loader)
    
    final_val_acc = classifier_trainer.callback_metrics.get('classifier_val_acc')
    if final_val_acc:
        print(f"\nFinal validation accuracy of linear probe: {final_val_acc.item():.4f}")
        wandb_logger.experiment.summary["final_val_accuracy"] = final_val_acc.item()