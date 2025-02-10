"""
Abusing dimensionality collapse for adversarial training
-------------------------------------------------------
We define:
- Generator G(noise) -> fake images
- RankNet D(x) -> embedding for each image in a batch,
                  measure SmoothRank across the batch.
Objectives:
- D wants: 
    * Real batch -> SmoothRank ~ 1
    * Fake batch -> SmoothRank ~ 0
- G wants:
    * Fake batch -> SmoothRank ~ 1 (fool D)

  real data => dimension-preserving,
  fake data => dimension-collapse.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# =====================
# 1) Hyperparams
# =====================
batch_size = 64
latent_dim = 32   # noise for generator
emb_dim    = 32   # dimension for D's embedding output
lr_g       = 1e-4
lr_d       = 1e-4
epochs     = 50

# SmoothRank margins
# For real batches, D wants SR ~ high_margin
# For fake batches, D wants SR ~ low_margin
high_margin = 0.8
low_margin  = 0.1

save_dir = "smoothrank_gan_demo"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# 2) Dataset (MNIST)
# =====================
transform = transforms.Compose([
    transforms.ToTensor(),
])
mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)


# =====================
# 3) Generator
# =====================
class Generator(nn.Module):
    """
    Simple MLP generator: (latent_dim -> 28x28).
    """
    def __init__(self, z_dim=100, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 28*28),
            nn.Sigmoid()  # produce [0,1] pixel
        )
    def forward(self, z):
        x = self.net(z)         # (B, 784)
        x = x.view(z.size(0), 1, 28, 28)
        return x


# =====================
# 4) RankNet (Discriminator)
# =====================
class RankNet(nn.Module):
    """
    Maps each image x -> an embedding in R^emb_dim,
    then we measure SmoothRank across the entire batch's embeddings.
    We'll combine with a margin-based objective:
      real batch => SR ~ 1
      fake batch => SR ~ 0
    """
    def __init__(self, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.GELU(),
            nn.Linear(256, emb_dim)
        )
    
    def embed(self, x):
        return self.net(x)  # (B, emb_dim)


# =====================
# 5) SmoothRank & Margins
# =====================
def smoothrank_value(latent: torch.Tensor, eps=1e-7) -> torch.Tensor:
    """
    SmoothRank in [0,1], differentiable w.r.t. latent.
    latent shape: (B, emb_dim)
    """
    s = torch.linalg.svdvals(latent)
    s_sum = s.sum() + eps
    p = s / s_sum
    H = -(p * (p+eps).log()).sum()
    B, E = latent.shape
    max_ent = math.log(min(B,E) + eps)
    return H / (max_ent + eps)

def sr_hinge_loss(latent: torch.Tensor, target_margin: float):
    sr = smoothrank_value(latent)
    diff = target_margin - sr
    loss = F.relu(diff) / (target_margin + 1e-9)
    return loss


# =====================
# 6) Instantiate G & D
# =====================
G = Generator(z_dim=latent_dim).to(device)
D = RankNet(emb_dim=emb_dim).to(device)

optim_G = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))


# =====================
# 7) Adversarial loop
# =====================
def d_loss_real(emb):
    """
    For real batch's embedding, we want SR >= high_margin
    => sr_hinge_loss(emb, high_margin).
    """
    return sr_hinge_loss(emb, target_margin=high_margin)

def d_loss_fake(emb):

    sr = smoothrank_value(emb)
    diff = sr - low_margin
    loss = F.relu(diff) / (low_margin + 1e-9)
    return loss

def g_loss_fake(emb):
    """
    G wants the fake batch to have SR >= high_margin
    => sr_hinge_loss(emb, high_margin).
    """
    return sr_hinge_loss(emb, target_margin=high_margin)

num_epochs = epochs
losses_D, losses_G = [], []

for epoch in range(1, num_epochs+1):
    for real_img, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
        real_img = real_img.to(device)
        B = real_img.size(0)

        # ============= Train D =============
        D.train()
        G.train()
        optim_D.zero_grad()
        
        # Real batch
        emb_real = D.embed(real_img)  # (B, emb_dim)
        d_real_loss = d_loss_real(emb_real)
        
        # Fake batch
        z_noise = torch.randn(B, latent_dim, device=device)
        x_fake  = G(z_noise)
        emb_fake = D.embed(x_fake.detach())
        d_fake_loss = d_loss_fake(emb_fake)
        
        d_total_loss = d_real_loss + d_fake_loss
        d_total_loss.backward()
        optim_D.step()
        
        # ============= Train G =============
        optim_G.zero_grad()
        
        # G tries to produce fake w/ SR ~ 1
        emb_fake_for_g = D.embed(x_fake)  # re-embed
        g_f_loss = g_loss_fake(emb_fake_for_g)
        g_f_loss.backward()
        optim_G.step()
        
        losses_D.append(d_total_loss.item())
        losses_G.append(g_f_loss.item())

    print(f"[Epoch {epoch}] D_loss={d_total_loss.item():.4f}, G_loss={g_f_loss.item():.4f}")

# =====================
# 8) Plot the losses
# =====================
plt.figure()
plt.plot(losses_D, label="D_loss")
plt.plot(losses_G, label="G_loss")
plt.legend()
plt.title("Rank-based GAN Losses")
plt.savefig(os.path.join(save_dir,"losses.png"))
plt.close()

# =====================
# 9) Sample from G
# =====================
G.eval()
with torch.no_grad():
    z_sample = torch.randn(16, latent_dim, device=device)
    x_gen = G(z_sample).cpu()
fig, axes = plt.subplots(4,4, figsize=(4,4))
axes = axes.flatten()
for i in range(16):
    axes[i].imshow(x_gen[i,0], cmap='gray')
    axes[i].axis('off')
plt.suptitle("Fake samples (Rank GAN?)")
plt.tight_layout()
plt.savefig(os.path.join(save_dir,"rank_gan_samples.png"))
plt.close()

print("Done! Check images in:", save_dir)
