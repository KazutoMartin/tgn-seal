import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

class TransformerLinkPred(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, max_z, num_heads=4, dropout=0.1):
        super().__init__()
        
        # 1. DRNL Structural Encoding
        self.z_embedding = nn.Embedding(max_z + 1, in_channels)
        
        # 2. Dense Transformer
        # batch_first=True is critical for implicitly triggering FlashAttention/SDPA
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels, 
            nhead=num_heads, 
            dim_feedforward=hidden_channels,
            dropout=dropout,
            batch_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Readout & MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, z, batch):
        """
        x: Temporal node features [N, in_channels]
        z: DRNL node labels [N]
        batch: Graph assignment vector [N]
        """
        # Inject DRNL structural features
        x = x + self.z_embedding(z)
        
        # Convert k-hop subgraphs into dense tensors
        # x_dense: [B, N_max, D], mask: [B, N_max] (True for valid nodes)
        x_dense, mask = to_dense_batch(x, batch)
        
        # PyTorch Transformer padding mask expects True for padded/ignored elements
        padding_mask = ~mask
        
        # Pass through Transformer. 
        # Device allocation is inherently safe here as padding_mask inherits the device from mask
        out_dense = self.transformer(x_dense, src_key_padding_mask=padding_mask)
        
        # Global Mean Pooling over valid nodes to get the subgraph representation
        mask_float = mask.unsqueeze(-1).float()
        pooled = (out_dense * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
        
        # Return link probability logits
        return self.mlp(pooled)