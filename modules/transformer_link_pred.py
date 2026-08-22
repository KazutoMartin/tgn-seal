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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        # 3. Readout & MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
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
        
        # --- DIRECTIONAL TARGET NODE EXTRACTION ---
        # Mask for Source Node (z == 1) and Destination Node (z == 2)
        src_mask = (z_dense == 1).unsqueeze(-1).float()
        dst_mask = (z_dense == 2).unsqueeze(-1).float()
        
        # Extract individual representations
        # Because there is exactly 1 src and 1 dst per subgraph, summing over dim=1 safely extracts the exact vector
        src_pooled = (out_dense * src_mask).sum(dim=1) 
        dst_pooled = (out_dense * dst_mask).sum(dim=1)
        
        # Concatenate the representations: [B, in_channels] + [B, in_channels] -> [B, 2 * in_channels]
        pooled = torch.cat([src_pooled, dst_pooled], dim=1)
        
        # Return link probability logits
        return self.mlp(pooled)