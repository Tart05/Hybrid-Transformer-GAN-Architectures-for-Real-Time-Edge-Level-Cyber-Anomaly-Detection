import torch
import torch.nn as nn

class TransformerGenerator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Map input features to a higher dimension for the Transformer
        self.embedding = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.reconstruction_head = nn.Linear(128, input_dim)

    def forward(self, x):
        # Transformer expects (Batch, Seq_Len, Features). We treat Seq_Len as 1.
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = self.reconstruction_head(x).squeeze(1)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)