# loss.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELossWithPseudo(nn.Module):
    """
    支持伪标签的 BCE 损失函数（按样本级别做加权平均）
    - 真实标签: target ∈ {-1, 0, +1}（其中 0=未知，不参与损失）
    - 伪标签位置: 由 pseudo_mask 指示（True 的位置按置信度加权）
    - 置信度: w = 2 * |sigmoid(logit) - 0.5| ∈ [0,1]，|logit| 越大越接近 1
    实现要点：对每个样本 n，
        numer_n = sum_c loss_n,c (real + pseudo*confidence)
        denom_n = sum_c (real_mask_n,c + pseudo_mask_n,c * confidence_n,c)
        sample_loss_n = numer_n / (denom_n + eps)
    最后对样本取平均或求和（由 size_average 控制）。
    """
    def __init__(self, margin=0.0, reduce=None, size_average=None):
        super(BCELossWithPseudo, self).__init__()
        self.margin = margin
        self.reduce = reduce
        self.size_average = size_average
        # 与项目保持一致：逐元素 loss，再根据掩码做聚合
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, input, target, pseudo_mask=None):
        """
        Args:
            input:  (N, C) 当前模型输出 logits
            target: (N, C) 标签 ∈ {-1, 0, +1}
            pseudo_mask: (N, C) bool，标记哪些位置是伪标签（只在这些位置做置信度加权）
        """
        input, target = input.float(), target.float()

        # 创建已知标签掩码（非未知标签的位置）
        known_mask = (target != 0)  # 已知标签包括 1 和 -1

        # 创建伪标签掩码
        pseudo_mask_float = pseudo_mask.float()

        # 真实标签掩码：已知标签且不是伪标签的位置
        real_mask = (known_mask & (~pseudo_mask)).float()

        # === 计算置信度权重（在 forward 中统一计算并 detach）
        # confidence = 2 * |sigmoid(input) - 0.5|，范围 [0,1]
        with torch.no_grad():
            pred_probs = torch.sigmoid(input)
            confidence_weights = 2 * torch.abs(pred_probs - 0.5)
            confidence_weights = torch.clamp(confidence_weights, 0.0, 1.0)
            confidence_weights = confidence_weights.detach()

        # === 真实标签损失 ===
        real_loss = self._original_bce_loss(input, target) * real_mask

        # === 伪标签损失（传入 confidence_weights） ===
        pseudo_loss = self._pseudo_bce_loss(input, target, pseudo_mask_float, confidence_weights)

        # 每元素损失矩阵 (N, C)
        loss_matrix = real_loss + pseudo_loss

        # === 按样本做加权平均（方案 A） ===
        if self.reduce:
            # per-element 权重矩阵：真实标签权重为1，伪标签权重为 confidence
            per_element_weight = real_mask + (pseudo_mask_float * confidence_weights)

            # numer & denom（按类累加，得到每个样本的值）
            numer = torch.sum(loss_matrix, dim=1)          # (N,)
            denom = torch.sum(per_element_weight, dim=1)   # (N,)

            eps = 1e-12
            sample_loss = numer / (denom + eps)

            # 将 denom == 0 的样本设为 0（无有效标签，不贡献损失）
            sample_loss = torch.where(denom > 0, sample_loss, torch.zeros_like(sample_loss))

            if self.size_average:
                return torch.mean(sample_loss)
            else:
                return torch.sum(sample_loss)

        # 不聚合则返回逐元素 loss 矩阵
        return loss_matrix

    def _original_bce_loss(self, input, target):
        """
        原始 BCE 损失（与工程中保持一致）：
        - 正标签：BCEWithLogitsLoss(input, 1)
        - 负标签：BCEWithLogitsLoss(-input, 1)
        未知（target==0）处掩掉（不参与）
        """
        positive_mask = (target > self.margin).float()
        negative_mask = (target < -self.margin).float()

        positive_loss = self.BCEWithLogitsLoss(input, torch.ones_like(input))
        negative_loss = self.BCEWithLogitsLoss(-input, torch.ones_like(input))

        loss = positive_mask * positive_loss + negative_mask * negative_loss
        return loss

    def _pseudo_bce_loss(self, input, target, pseudo_mask, confidence_weights):
        """
        伪标签的损失计算（接受外部传入的置信度矩阵）
        - positive_pseudo_mask / negative_pseudo_mask: 仍沿用原工程正/负计算方式
        - 伪标签位置的损失乘以 confidence_weights
        返回形状 (N, C)
        """
        # 伪标签的正/负掩码
        positive_pseudo_mask = ((target > self.margin).float() * pseudo_mask)
        negative_pseudo_mask = ((target < -self.margin).float() * pseudo_mask)

        # 复用原始写法：正伪标签希望 input -> +∞，负伪标签希望 input -> -∞
        pos_raw = self.BCEWithLogitsLoss(input, torch.ones_like(input))
        neg_raw = self.BCEWithLogitsLoss(-input, torch.ones_like(input))

        # 仅在伪标签位置生效，并乘以置信度权重
        pos_loss = pos_raw * (positive_pseudo_mask * confidence_weights)
        neg_loss = neg_raw * (negative_pseudo_mask * confidence_weights)

        return pos_loss + neg_loss


class BCELoss(nn.Module):
    """
    原始的 BCE 损失函数（用于纯真实标签阶段）
    """
    def __init__(self, margin=0.0, reduce=None, size_average=None):
        super(BCELoss, self).__init__()
        self.margin = margin
        self.reduce = reduce
        self.size_average = size_average
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, input, target):
        input, target = input.float(), target.float()
        positive_mask = (target > self.margin).float()
        negative_mask = (target < -self.margin).float()

        # 与原工程保持一致的“对称 BCE”写法
        positive_loss = self.BCEWithLogitsLoss(input, target)
        negative_loss = self.BCEWithLogitsLoss(-input, -target)
        loss = positive_mask * positive_loss + negative_mask * negative_loss

        if self.reduce:
            valid_mask = ((positive_mask > 0) | (negative_mask > 0))
            if self.size_average:
                return torch.mean(loss[valid_mask]) if valid_mask.any() else torch.mean(loss)
            else:
                return torch.sum(loss[valid_mask]) if valid_mask.any() else torch.sum(loss)
        return loss
