import torch
from torch import nn

class LossFn(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.11):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, out):
        if out["rpn_reg_input"].numel() == 0:
            rpn_l1 = torch.ones((1)) * 0
        else:
            rpn_l1 = nn.functional.smooth_l1_loss(
                out["rpn_reg_input"], out["rpn_reg_target"], beta=self.beta)
        rpn_cls = nn.functional.binary_cross_entropy_with_logits(
            out["rpn_cls_input"], out["rpn_cls_target"])
        if (out["roi_reg_input"] is None) or (out["roi_reg_target"] is None):
            roi_l1 = torch.ones((1)) * 0
        else:
            roi_l1 = nn.functional.smooth_l1_loss(
                out["roi_reg_input"], out["roi_reg_target"], beta=self.beta)
        roi_cls = nn.functional.cross_entropy(out["roi_cls_input"], out["roi_cls_target"])
        rpn_loss = rpn_l1 + rpn_cls
        roi_loss = roi_l1 + roi_cls
        loss = rpn_loss + roi_loss
        return {
            "total_loss": loss,
            "rpn_loss": rpn_loss,
            "roi_loss": roi_loss,
        }