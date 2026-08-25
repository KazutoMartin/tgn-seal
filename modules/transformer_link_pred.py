import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

class TransformerLinkPred(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, max_z, num_heads=4, dropout=0.1, pooling_type="mean"):
        super().__init__()
        
        self.pooling_type = pooling_type
        
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        # 3. Readout & MLP
        # Dynamically calculate the input size based on the pooling choice
        mlp_in_channels = in_channels * 2 if pooling_type == "target" else in_channels
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_channels, hidden_channels),
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
        
        # Convert k-hop subgraphs and labels into dense tensors
        x_dense, mask = to_dense_batch(x, batch)
        z_dense, _ = to_dense_batch(z, batch)
        
        # PyTorch Transformer padding mask expects True for padded/ignored elements
        padding_mask = ~mask
        
        # Pass through Transformer
        out_dense = self.transformer(x_dense, src_key_padding_mask=padding_mask)
        
        if self.pooling_type == "target":
            # --- DIRECTIONAL TARGET NODE EXTRACTION ---
            # Mask for Source Node (z == 1) and Destination Node (z == 2)
            src_mask = (z_dense == 1).unsqueeze(-1).float()
            dst_mask = (z_dense == 2).unsqueeze(-1).float()
            
            src_pooled = (out_dense * src_mask).sum(dim=1) 
            dst_pooled = (out_dense * dst_mask).sum(dim=1)
            
            # Concatenate the representations: [B, in_channels] + [B, in_channels] -> [B, 2 * in_channels]
            pooled = torch.cat([src_pooled, dst_pooled], dim=1)
            
        else:
            # --- MEAN POOLING (DEFAULT) ---
            mask_float = mask.unsqueeze(-1).float()
            pooled = (out_dense * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
        
        # Return link probability logits
        return self.mlp(pooled)