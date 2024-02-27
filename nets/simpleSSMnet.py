import torch
from torch import nn
import pytorch_lightning as pl
import sys
sys.path.append('..')
from utils.SSMblock import SSMblock  # Assuming your SSMblock class is defined in a file named ssm_block.py

class MyLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.ssm_block1 = SSMblock(P=32, H=1) # P -> latent dimension | H -> channels
        self.ssm_block2 = SSMblock(P=16, H=1) # P -> latent dimension | H -> channels
        self.linear = nn.Linear(1, 10)
        
    def forward(self, x):
        x = self.ssm_block1(x)
        x = self.ssm_block2(x)
        x = self.linear(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
