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
# 0. Experiment Configuration
# --------------------------------------------------------------------------
# Controls the variance loss.
BIDIRECTIONAL_VARIANCE_LOSS = False
VAR_SAMPLE_FACTOR = 1.0
# VAR_TARGET_INIT = lambda num_features: torch.rand(num_features)*VAR_SAMPLE_FACTOR*5
VAR_TARGET_INIT = lambda num_features: torch.ones(num_features)*VAR_SAMPLE_FACTOR

LAYER_DIMS = [(28 * 28, 512), (512, 256)]
INVARIANCE_L1 = False

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
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

# --------------------------------------------------------------------------
# 2. Semi-Local Loss with Random Variance Targets
# --------------------------------------------------------------------------

class RandomTargetVarianceLoss(nn.Module):
    """
    A semi-local VICReg loss where each neuron has its own random variance target.
    """
    def __init__(self, num_features, sim_coeff=25.0, std_coeff=25.0, cov_coeff=5000.0, bidirectional=False):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.bidirectional = bidirectional

        # Create a persistent buffer for the random targets.
        # This ensures they are fixed and moved to the correct device.
        # Targets are sampled from U(0, 1).
        self.register_buffer("variance_targets", VAR_TARGET_INIT(num_features))

    def forward(self, z_a, z_b):
        z_b = z_b.detach()
        num_features = z_a.shape[1]

        if not INVARIANCE_L1:
            # 1. Neuron-wise Invariance Loss (local signal)
            sim_loss_per_neuron = F.mse_loss(z_a, z_b, reduction='none').mean(dim=0)
        else:
            sim_loss_per_neuron = F.l1_loss(z_a, z_b, reduction='none').mean(dim=0)

        # 2. Neuron-wise Variance Loss with random targets (local signal)
        std_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
        
        if self.bidirectional:
            # Punish deviation from target in both directions
            std_loss_per_neuron = torch.abs(std_a - self.variance_targets)
        else:
            # Only punish having less variance than the random target
            std_loss_per_neuron = F.relu(self.variance_targets - std_a)

        # Sum the local losses across all neurons to get a single scalar
        local_loss = (self.sim_coeff * sim_loss_per_neuron +
                      self.std_coeff * std_loss_per_neuron).sum()

        # 3. Global Covariance Loss (non-local signal)
        z_a_centered = z_a - z_a.mean(dim=0)
        cov_a = (z_a_centered.T @ z_a_centered) / (z_a.shape[0] - 1)
        cov_loss = off_diagonal(cov_a).pow_(2).sum() / num_features

        # Combine local and global parts
        total_loss = local_loss + self.cov_coeff * cov_loss
        loss_dict = {
            'sim': sim_loss_per_neuron.sum(),
            'std': std_loss_per_neuron.sum(),
            'cov': cov_loss
        }
        return total_loss, loss_dict

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# --------------------------------------------------------------------------
# 3. Modular MLP Architecture (Unchanged)
# --------------------------------------------------------------------------

class GreedyMLPEncoder(nn.Module):
    """An MLP encoder composed of a list of modules, trained greedily."""
    def __init__(self, layer_dims=[(28*28, 512), (512, 256)]):
        super().__init__()
        self.rep_dim = layer_dims[-1][1]
        self.layers = nn.ModuleList()
        for in_dim, out_dim in layer_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU()
            ))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

# --------------------------------------------------------------------------
# 4. Greedy Layer-Wise Unsupervised Trainer
# --------------------------------------------------------------------------

class LayerWiseVICRegTrainer(pl.LightningModule):
    def __init__(self, epochs_per_layer=5, lr=1e-3, bidirectional_loss=False):
        super().__init__()
        self.save_hyperparameters()
        
        # Define MLP architecture and instantiate encoder
        self.layer_dims = LAYER_DIMS
        self.encoder = GreedyMLPEncoder(layer_dims=self.layer_dims)
        
        # Create a list of loss functions, one for each layer
        self.criterions = nn.ModuleList()
        for _, out_dim in self.layer_dims:
            self.criterions.append(
                RandomTargetVarianceLoss(num_features=out_dim, bidirectional=bidirectional_loss)
            )
        
        self.current_layer_idx = 0
        self.epochs_per_layer = epochs_per_layer
        self.automatic_optimization = False

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        opt = optimizers[self.current_layer_idx]
        criterion = self.criterions[self.current_layer_idx]
        (x1, x2), _ = batch
        
        inp1 = x1.view(x1.size(0), -1)
        inp2 = x2.view(x2.size(0), -1)
        
        with torch.no_grad():
            for i in range(self.current_layer_idx):
                self.encoder.layers[i].eval()
                inp1 = self.encoder.layers[i](inp1)
                inp2 = self.encoder.layers[i](inp2)
        
        current_layer = self.encoder.layers[self.current_layer_idx]
        current_layer.train()
        y1 = current_layer(inp1)
        y2 = current_layer(inp2)

        loss, loss_dict = criterion(y1, y2)
        self.log(f'layer_{self.current_layer_idx}/vicreg_loss', loss, prog_bar=True)
        self.log_dict({f'layer_{self.current_layer_idx}/loss_{k}': v for k, v in loss_dict.items()})
        
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

    def configure_optimizers(self):
        return [torch.optim.Adam(layer.parameters(), lr=self.hparams.lr) for layer in self.encoder.layers]

    def on_train_epoch_end(self):
        if self.trainer.current_epoch > 0 and \
           (self.trainer.current_epoch + 1) % self.epochs_per_layer == 0:
            if self.current_layer_idx < len(self.encoder.layers) - 1:
                self.current_layer_idx += 1
                print(f"\n\n--- Switching to train layer {self.current_layer_idx} ---")

# --------------------------------------------------------------------------
# 5. Supervised Linear Classifier (Unchanged)
# --------------------------------------------------------------------------

class LinearClassifier(pl.LightningModule):
    def __init__(self, encoder, lr=1e-3, num_classes=10, binarize_inputs=False):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.encoder.rep_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
    
    def set_binarize_inputs(self, binarize):
        self.hparams.binarize_inputs = binarize

    def forward(self, x):
        with torch.no_grad():
            reps = self.encoder(x)
        if self.hparams.binarize_inputs:
            reps = (reps > 0).float()
        return self.classifier(reps)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('classifier_train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('classifier_val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.lr)

# --------------------------------------------------------------------------
# 6. Execution
# --------------------------------------------------------------------------

BATCH_SIZE = 256
EPOCHS_PER_LAYER = 40
CLASSIFIER_EPOCHS = 20
DATA_PATH = './data'

greedy_trainer_model_builder = lambda: LayerWiseVICRegTrainer(
    epochs_per_layer=EPOCHS_PER_LAYER, lr=1e-3, bidirectional_loss=BIDIRECTIONAL_VARIANCE_LOSS
)

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

train_loader_builder = lambda:DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=True
)

val_loader_builder = lambda: DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True
)


if __name__ == '__main__':
    pl.seed_everything(42)
    
    train_loader = train_loader_builder()
    val_loader = val_loader_builder()
    wandb_logger = WandbLogger(project="experiments_mnist", name=f"vicreg_mlp_random_targets_bidir_{BIDIRECTIONAL_VARIANCE_LOSS}")
    
    greedy_trainer_model = greedy_trainer_model_builder()
    
    num_layers = len(greedy_trainer_model.encoder.layers)
    total_pretrain_epochs = num_layers * EPOCHS_PER_LAYER
    
    pre_trainer = pl.Trainer(
        max_epochs=total_pretrain_epochs, accelerator="auto", devices=1,
        logger=wandb_logger, enable_checkpointing=False
    )

    print(f"--- Starting Greedy Layer-Wise Pre-training (Random Targets, Bidirectional={BIDIRECTIONAL_VARIANCE_LOSS}) ---")
    pre_trainer.fit(greedy_trainer_model, train_loader)
    print("--- Pre-training Finished ---")
    
    print("\n--- Starting Supervised Linear Evaluation ---")
    frozen_encoder = greedy_trainer_model.encoder
    classifier_model = LinearClassifier(encoder=frozen_encoder, lr=1e-3)
    
    classifier_trainer = pl.Trainer(
        max_epochs=CLASSIFIER_EPOCHS, accelerator="auto", devices=1,
        logger=wandb_logger, enable_checkpointing=False, num_sanity_val_steps=0
    )
    classifier_trainer.fit(classifier_model, val_loader)
    print("--- Training of linear probe finished ---")
    print("--- Running final validation on test set ---")
    classifier_trainer.validate(classifier_model, dataloaders=val_loader)
    
    final_val_acc = classifier_trainer.callback_metrics.get('classifier_val_acc')
    if final_val_acc:
        print(f"\nFinal validation accuracy of linear probe: {final_val_acc.item():.4f}")
        wandb_logger.experiment.summary["final_val_accuracy"] = final_val_acc.item()

    # save the encoder
    os.makedirs("saved_models", exist_ok=True)
    torch.save(frozen_encoder.state_dict(), f"saved_models/vicreg_5_nl_encoder_{BIDIRECTIONAL_VARIANCE_LOSS}.pth")
    torch.save(greedy_trainer_model.hparams, f"saved_models/vicreg_5_nl__hparams_{BIDIRECTIONAL_VARIANCE_LOSS}.pth")
    torch.save(classifier_model.state_dict(), f"saved_models/vicreg_5_nl_classifier_{BIDIRECTIONAL_VARIANCE_LOSS}.pth")
    print("Saved trained models to 'saved_models/' directory.")