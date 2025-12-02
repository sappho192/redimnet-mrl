"""
Loss functions for Matryoshka Representation Learning.

Includes MatryoshkaLoss wrapper and AAMSoftmax (ArcFace) for speaker verification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MatryoshkaLoss(nn.Module):
    """
    Multi-resolution contrastive loss for MRL training.

    Applies base loss (e.g., AAM-Softmax, SubCenter, Triplet)
    at multiple embedding dimensions simultaneously.

    Args:
        base_loss: Base loss module (e.g., AAMSoftmax, CrossEntropyLoss)
        mrl_dims: List of MRL dimensions to train on
        mrl_weights: Optional weights for each dimension (default: equal weighting)
        normalize_embeddings: Whether to L2-normalize embeddings before loss

    Example:
        >>> base_loss = AAMSoftmax(embed_dim=256, num_classes=5994)
        >>> criterion = MatryoshkaLoss(
        ...     base_loss=base_loss,
        ...     mrl_dims=[64, 128, 192, 256],
        ...     mrl_weights=[1.0, 1.0, 1.0, 1.0]
        ... )
        >>> embeddings_dict = model(audio, return_all_dims=True)
        >>> loss, loss_dict = criterion(embeddings_dict, labels)
    """
    def __init__(
        self,
        base_loss,
        mrl_dims,
        mrl_weights=None,
        normalize_embeddings=True,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.mrl_dims = sorted(mrl_dims)
        self.normalize_embeddings = normalize_embeddings

        # Default: equal weighting
        if mrl_weights is None:
            mrl_weights = [1.0] * len(mrl_dims)
        assert len(mrl_weights) == len(mrl_dims), \
            "mrl_weights must have same length as mrl_dims"

        self.register_buffer('mrl_weights', torch.tensor(mrl_weights))

    def forward(self, embeddings_dict, labels):
        """
        Compute weighted MRL loss.

        Args:
            embeddings_dict: Dictionary {dim: tensor[B, dim]} from model
            labels: Tensor[B] with speaker labels

        Returns:
            total_loss: Weighted sum of losses at each dimension
            loss_dict: Individual losses for logging
        """
        total_loss = 0
        loss_dict = {}

        for i, dim in enumerate(self.mrl_dims):
            emb = embeddings_dict[dim]

            # Normalize embeddings if required
            if self.normalize_embeddings:
                emb = F.normalize(emb, p=2, dim=1)

            # Compute loss for this dimension
            loss = self.base_loss(emb, labels)

            # Weight and accumulate
            weight = self.mrl_weights[i]
            total_loss += weight * loss

            # Log individual losses
            loss_dict[f'loss_{dim}d'] = loss.item()
            loss_dict[f'weight_{dim}d'] = weight.item()

        # Normalize by total weight
        total_weight = self.mrl_weights.sum()
        total_loss = total_loss / total_weight

        loss_dict['loss_total'] = total_loss.item()
        loss_dict['total_weight'] = total_weight.item()

        return total_loss, loss_dict


class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax (ArcFace) for speaker verification.

    Reference:
        Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
        CVPR 2019

    Args:
        embed_dim: Embedding dimension
        num_classes: Number of speaker classes
        margin: Angular margin (default: 0.2)
        scale: Feature scale (default: 30)
        easy_margin: Use easy margin variant (default: False)

    Example:
        >>> loss_fn = AAMSoftmax(embed_dim=256, num_classes=5994, margin=0.2, scale=30)
        >>> embeddings = torch.randn(32, 256)  # [batch_size, embed_dim]
        >>> labels = torch.randint(0, 5994, (32,))
        >>> loss = loss_fn(embeddings, labels)
    """
    def __init__(
        self,
        embed_dim,
        num_classes,
        margin=0.2,
        scale=30.0,
        easy_margin=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin

        # Weight matrix: [num_classes, embed_dim]
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        # Cosine margin parameters
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        """
        Compute AAM-Softmax loss.

        Args:
            embeddings: Normalized embeddings [B, embed_dim]
            labels: Ground truth labels [B]

        Returns:
            loss: Scalar loss value
        """
        # Normalize embeddings and weights
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        weight_normalized = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: [B, num_classes]
        cosine = F.linear(embeddings_normalized, weight_normalized)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Calculate sine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Compute cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding for target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        # Compute cross-entropy loss
        loss = F.cross_entropy(output, labels)

        return loss


class SubCenterAAMSoftmax(nn.Module):
    """
    Sub-Center Additive Angular Margin Softmax.

    Extends AAMSoftmax with multiple sub-centers per class to handle
    intra-class variance in speaker embeddings.

    Reference:
        Deng et al. "Sub-center ArcFace: Boosting Face Recognition by
        Large-Scale Noisy Web Faces" ECCV 2020

    Args:
        embed_dim: Embedding dimension
        num_classes: Number of speaker classes
        num_subcenters: Number of sub-centers per class (default: 3)
        margin: Angular margin (default: 0.2)
        scale: Feature scale (default: 30)
    """
    def __init__(
        self,
        embed_dim,
        num_classes,
        num_subcenters=3,
        margin=0.2,
        scale=30.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters
        self.margin = margin
        self.scale = scale

        # Sub-center weight matrix: [num_classes * num_subcenters, embed_dim]
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes * num_subcenters, embed_dim)
        )
        nn.init.xavier_uniform_(self.weight)

        # Margin parameters
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        """
        Compute Sub-Center AAM-Softmax loss.

        Args:
            embeddings: Normalized embeddings [B, embed_dim]
            labels: Ground truth labels [B]

        Returns:
            loss: Scalar loss value
        """
        batch_size = embeddings.size(0)

        # Normalize
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        weight_normalized = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity with all sub-centers: [B, num_classes * num_subcenters]
        cosine = F.linear(embeddings_normalized, weight_normalized)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Reshape to separate sub-centers: [B, num_classes, num_subcenters]
        cosine = cosine.view(batch_size, self.num_classes, self.num_subcenters)

        # Select maximum cosine per class (closest sub-center)
        cosine, _ = torch.max(cosine, dim=2)  # [B, num_classes]

        # Calculate sine and phi (same as AAMSoftmax)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding and margin application
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        # Cross-entropy loss
        loss = F.cross_entropy(output, labels)

        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.

    Args:
        margin: Margin for triplet loss (default: 0.3)
        mining_strategy: 'hard', 'semi-hard', or 'all' (default: 'hard')

    Example:
        >>> criterion = TripletLoss(margin=0.3, mining_strategy='hard')
        >>> embeddings = model(audio)  # [B, D]
        >>> labels = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> loss = criterion(embeddings, labels)
    """
    def __init__(self, margin=0.3, mining_strategy='hard'):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy

    def forward(self, embeddings, labels):
        """
        Compute triplet loss with hard negative mining.

        Args:
            embeddings: Embeddings [B, D]
            labels: Speaker labels [B]

        Returns:
            loss: Scalar loss value
        """
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for positive and negative pairs
        labels = labels.unsqueeze(1)
        mask_positive = (labels == labels.t()).float()
        mask_negative = (labels != labels.t()).float()

        # Remove diagonal (self-comparison)
        mask_positive = mask_positive - torch.eye(
            mask_positive.size(0), device=mask_positive.device
        )

        if self.mining_strategy == 'hard':
            # Hard positive: farthest positive
            distances_positive = distances * mask_positive
            hardest_positive, _ = distances_positive.max(dim=1)

            # Hard negative: closest negative
            distances_negative = distances + 1e6 * (1 - mask_negative)
            hardest_negative, _ = distances_negative.min(dim=1)

            # Triplet loss
            loss = F.relu(hardest_positive - hardest_negative + self.margin)
            loss = loss.mean()

        elif self.mining_strategy == 'all':
            # All valid triplets
            num_samples = embeddings.size(0)
            losses = []

            for i in range(num_samples):
                # Positive samples for anchor i
                positive_mask = mask_positive[i].bool()
                if positive_mask.sum() == 0:
                    continue

                # Negative samples for anchor i
                negative_mask = mask_negative[i].bool()
                if negative_mask.sum() == 0:
                    continue

                # Compute all triplet losses
                pos_dists = distances[i][positive_mask]
                neg_dists = distances[i][negative_mask]

                # Broadcasting: [num_pos, num_neg]
                triplet_losses = F.relu(
                    pos_dists.unsqueeze(1) - neg_dists.unsqueeze(0) + self.margin
                )
                losses.append(triplet_losses.mean())

            if len(losses) > 0:
                loss = torch.stack(losses).mean()
            else:
                loss = torch.tensor(0.0, device=embeddings.device)

        else:
            raise ValueError(f"Unknown mining strategy: {self.mining_strategy}")

        return loss


if __name__ == "__main__":
    # Test loss functions
    print("Testing MRL loss functions...")

    # Test AAMSoftmax
    print("\n1. Testing AAMSoftmax:")
    aam_loss = AAMSoftmax(embed_dim=256, num_classes=100, margin=0.2, scale=30)
    embeddings = F.normalize(torch.randn(16, 256), p=2, dim=1)
    labels = torch.randint(0, 100, (16,))
    loss = aam_loss(embeddings, labels)
    print(f"   Loss: {loss.item():.4f}")

    # Test MatryoshkaLoss
    print("\n2. Testing MatryoshkaLoss:")
    base_loss = AAMSoftmax(embed_dim=256, num_classes=100)
    mrl_loss = MatryoshkaLoss(
        base_loss=base_loss,
        mrl_dims=[64, 128, 192, 256],
        mrl_weights=[1.0, 1.0, 1.0, 1.0]
    )

    # Simulate multi-dimension embeddings
    emb_dict = {
        64: F.normalize(torch.randn(16, 64), p=2, dim=1),
        128: F.normalize(torch.randn(16, 128), p=2, dim=1),
        192: F.normalize(torch.randn(16, 192), p=2, dim=1),
        256: F.normalize(torch.randn(16, 256), p=2, dim=1),
    }
    total_loss, loss_dict = mrl_loss(emb_dict, labels)
    print(f"   Total Loss: {total_loss.item():.4f}")
    for key, val in loss_dict.items():
        print(f"   {key}: {val:.4f}")

    # Test TripletLoss
    print("\n3. Testing TripletLoss:")
    triplet_loss = TripletLoss(margin=0.3, mining_strategy='hard')
    embeddings = F.normalize(torch.randn(16, 128), p=2, dim=1)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    loss = triplet_loss(embeddings, labels)
    print(f"   Loss: {loss.item():.4f}")

    print("\n✅ All loss tests passed!")
