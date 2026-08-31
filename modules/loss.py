import torch
import torch.nn as nn
import torch.nn.functional as F

class FastConvergenceFocalLoss(nn.Module):
    """Numerically stable binary Focal Loss with configurable alpha and gamma.
    
    Modulates standard BCE with (1 - p_t)^gamma to down-weight easy negatives 
    and maintain dense gradients on hard 2-hop negative boundaries.
    """
    def __init__(self, alpha=0.5, gamma=1.2, reduction="mean"):
        super(FastConvergenceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        # Calculate p_t
        p = torch.sigmoid(inputs)
        pt = p * targets + (1.0 - p) * (1.0 - targets)
        
        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1.0 - pt) ** self.gamma
        
        # Alpha class balancing
        if self.alpha is not None and self.alpha >= 0:
            alpha_weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = alpha_weight * focal_weight
            
        loss = focal_weight * bce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

