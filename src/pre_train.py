
import torch
import pickle
import random
import json
import numpy as np

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from matplotlib import pyplot as plt

from data import *
from tokenizer import *
from utils import *
from model import *

epochs = 100

dataloader = DataLoader(dataset, batch_size = 192, shuffle = True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net(len(tokenizer.key_value), n_layers = 6)
model = model.apply(init_weights)
model = model.to(device)

max_lr = 5e-04
loss_fn = nn.CrossEntropyLoss()

opt = AdamW(model.parameters(), lr = max_lr, weight_decay = 1e-02)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                   max_lr,
                                                   epochs=epochs,
                                                   steps_per_epoch=len(dataloader),
                                                   pct_start=0.1,
                                                   anneal_strategy='cos',
                                                   cycle_momentum=True,
                                                   base_momentum=0.85,
                                                   max_momentum=0.95,
                                                   div_factor=100.0,
                                                   final_div_factor=1000,
                                                   last_epoch=-1)

# variances = np.linspace(0.05, 0.15, epochs)
train_losses = []
for epoch in range(epochs):
    train_iter = iter(dataloader)
    for it, (x, y) in enumerate(train_iter):

        tokens, mask = x[0].to(device), x[1].to(device)
        y = y.to(torch.long).to(device)

        logits = model(tokens, mask)
        loss = loss_fn(logits, y)

        train_losses.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
        lr_scheduler.step()

    print(f"Epoch: {epoch} | Train Loss: {np.mean(train_losses[-len(dataloader):]):.4f}")
    if epoch % 5 == 0: # do checkpointing
        # PLS FINISH THE CODE
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': np.mean(train_losses[-len(dataloader):])
        }
        torch.save(checkpoint, f'/content/drive/MyDrive/hrd_hack/model_checkpoint_epoch_{epoch}.pt')
        print(f"Checkpoint saved at epoch {epoch}")
