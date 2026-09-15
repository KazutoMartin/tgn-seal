import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
import numpy as np

class TransformerLinkPred(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, max_z, num_heads=4, dropout=0.1, pooling_type="mean", use_temporal_decay=False, lambda_decay=0.9, decay_scale=86400.0):
        super().__init__()
        
        self.pooling_type = pooling_type

        self.use_temporal_decay = use_temporal_decay
        self.lambda_decay = lambda_decay
        self.decay_scale = decay_scale
        
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

    def forward(self, x, z, batch, edge_index, node_timestamps=None, query_time=None):
        z = torch.clamp(z, max=self.z_embedding.num_embeddings - 1)

        # Inject DRNL structural features
        x = x + self.z_embedding(z)

        if self.use_temporal_decay and node_timestamps is not None and query_time is not None:
            # PyG DataLoader does not auto-concatenate NumPy arrays, returning a list instead.
            if isinstance(node_timestamps, list):
                # Concatenate, cast to float32 (to prevent precision mismatches), and move to GPU
                node_timestamps = torch.from_numpy(np.concatenate(node_timestamps)).float().to(x.device)
            else:
                node_timestamps = node_timestamps.to(x.device)
            subgraph_query_times = query_time.squeeze(-1) if query_time.ndim > 1 else query_time
            node_query_times = subgraph_query_times[batch]
            
            delta_t = torch.clamp(node_query_times - node_timestamps, min=0.0)
            delta_t_scaled = delta_t / max(self.decay_scale, 1e-6)
            
            decay_weights = (self.lambda_decay ** delta_t_scaled).unsqueeze(-1)
            x = x * decay_weights
        
        # Convert k-hop subgraphs and labels into dense tensors
        x_dense, mask = to_dense_batch(x, batch)
        z_dense, _ = to_dense_batch(z, batch)
        
        # --- NEW: Compute Subgraph Density ---
        # Number of nodes per subgraph |V_i|
        v_i = mask.sum(dim=1).float()
        # Number of edges per subgraph |E_i|
        e_i = torch.bincount(batch[edge_index[0]], minlength=x_dense.shape[0]).float()
        # Local Subgraph Edge Density
        density = e_i / (v_i * (v_i - 1.0)).clamp(min=1e-9)
        
        # PyTorch Transformer padding mask expects True for padded/ignored elements
        padding_mask = ~mask
        
        # Pass through Transformer
        out_dense = self.transformer(x_dense, src_key_padding_mask=padding_mask)
        
        if self.pooling_type == "target":
            # default
            src_mask = (z_dense == 1).unsqueeze(-1).float()
            dst_mask = (z_dense == 2).unsqueeze(-1).float()
            
            src_pooled = (out_dense * src_mask).sum(dim=1) 
            dst_pooled = (out_dense * dst_mask).sum(dim=1)
            pooled = torch.cat([src_pooled, dst_pooled], dim=1)
            
        else:
            mask_float = mask.unsqueeze(-1).float()
            pooled = (out_dense * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
        
        # Return link probability logits AND density
        return self.mlp(pooled), density