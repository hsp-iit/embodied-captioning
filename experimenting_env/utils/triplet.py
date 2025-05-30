"""
Credits https://github.com/KinWaiCheuk/pytorch-triplet-loss/blob/master/TNN/Mining.py

This code is a re-implmentationm of the code written in Tensorflow
https://omoindrot.github.io/triplet-loss
https://github.com/omoindrot/tensorflow-triplet-loss

All the doc strings and comments are from the original implmentation
"""

import os
import random
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def offline(x, y, testsize=0.3, ap_pairs=10, an_pairs=10):
    data_xy = tuple([x, y])

    trainsize = 1 - testsize

    triplet_train_pairs = []
    triplet_test_pairs = []
    for data_class in sorted(set(data_xy[1])):

        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]
        A_P_pairs = random.sample(
            list(permutations(same_class_idx, 2)), k=ap_pairs
        )  # Generating Anchor-Positive pairs
        Neg_idx = random.sample(list(diff_class_idx), k=an_pairs)

        # train
        A_P_len = len(A_P_pairs)
        # Neg_len = len(Neg_idx)
        for ap in A_P_pairs[: int(A_P_len * trainsize)]:
            Anchor = data_xy[0][ap[0]]
            Positive = data_xy[0][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                triplet_train_pairs.append([Anchor, Positive, Negative])
        # test
        for ap in A_P_pairs[int(A_P_len * trainsize) :]:
            Anchor = data_xy[0][ap[0]]
            Positive = data_xy[0][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                triplet_test_pairs.append([Anchor, Positive, Negative])

    return np.array(triplet_train_pairs), np.array(triplet_test_pairs)


def _get_triplet_mask(labels, device='cpu'):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: torch.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size()[0]).type(
        torch.ByteTensor
    )  # make a index in Bool
    indices_not_equal = -(indices_equal - 1)  # flip booleans
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)

    valid_labels = i_equal_j & ~i_equal_k
    # Combine the two masks

    mask = distinct_indices.to(device).bool() & valid_labels

    return mask


def _pairwise_distances(embeddings, squared=False, device='cpu'):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    embeddings = torch.squeeze(embeddings, 1)  # shape=(batch_size, features, 1)
    dot_product = torch.matmul(
        embeddings, torch.transpose(embeddings, 0, 1)
    )  # shape=(batch_size, batch_size)

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)
    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = (
        torch.unsqueeze(square_norm, 1)
        - 2.0 * dot_product
        + torch.unsqueeze(square_norm, 0)
    )

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.max(distances, torch.Tensor([0.0]).to(device))
    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)
    # return distances matrix (batch X batch)
    return distances


def online_mine_all(labels, embeddings, margin, squared=False, device='cpu'):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared, device=device)
    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels, device=device)
    mask = mask.float()

    triplet_loss = mask * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = torch.max(triplet_loss, torch.Tensor([0.0]).to(device))

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = (triplet_loss > 1e-16).float()

    num_positive_triplets = torch.sum(valid_triplets)

    num_valid_triplets = torch.sum(mask)
    # fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, num_positive_triplets, num_valid_triplets


def _get_anchor_positive_triplet_mask(labels, device):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size()[0]).bool().to(device)

    indices_not_equal = ~indices_equal  # flip booleans

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    # Combine the two masks
    mask = indices_not_equal & labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels, device):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    mask = ~labels_equal  # invert the boolean tensor

    return mask


def online_mine_hard(labels, embeddings, margin=0.2, squared=False, device='cpu'):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared, device=device)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels, device)
    mask_anchor_positive = mask_anchor_positive

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist = torch.max(anchor_positive_dist, 1, keepdim=True)[0]

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels, device)
    mask_anchor_negative = mask_anchor_negative

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = torch.max(pairwise_dist, 1, keepdim=True)[0]
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * ~(
        mask_anchor_negative
    )

    # shape (batch_size,)
    hardest_negative_dist = torch.min(anchor_negative_dist, 1, keepdim=True)[0]

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = torch.max(
        hardest_positive_dist - hardest_negative_dist + margin,
        torch.Tensor([0.0]).to(device),
    )

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return (
        triplet_loss,
        torch.sum(mask_anchor_positive),
        torch.sum(mask_anchor_negative),
    )


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, labels, features, mask=None):

        device = features.device

        if len(features.shape) < 2:
            raise ValueError(
                '`features` needs to be [bsz, n_views, ...],'
                'at least 3 dimensions are required'
            )

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # compute logits
        anchors = torch.div(torch.matmul(features, features.T), self.temperature)

        # for numerical stability

        logits_max, _ = torch.max(anchors, dim=1, keepdim=True)
        logits = anchors - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()

        return loss


def simsiam(p, z):

    return -F.cosine_similarity(p, z.detach(), dim=-1).mean()


def compute_splits(labels):
    batch_size = len(labels)
    device = labels.device
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels, device)
    split1 = torch.zeros_like(labels)
    split2 = torch.zeros_like(labels)

    for idx in range(batch_size):
        x = mask_anchor_positive[idx]
        if x.sum() == 0:
            continue
        s = torch.multinomial(x / x.sum(), 1)
        split1[idx] = True
        split2[s] = True

        mask_anchor_positive[:, s] = False
        mask_anchor_positive[:, idx] = False
        mask_anchor_positive[s, :] = False
    return split1.bool(), split2.bool()


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
                                    Projection MLP. The projection MLP (in f) has BN ap-
                                    plied to each fully-connected (fc) layer, including its out-
                                    put fc. Its output fc has no ReLU. The hidden fc is 2048-d.
                                    This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class tinyprojection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
                                    Projection MLP. The projection MLP (in f) has BN ap-
                                    plied to each fully-connected (fc) layer, including its out-
                                    put fc. Its output fc has no ReLU. The hidden fc is 2048-d.
                                    This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        x = self.layer1(x)

        return F.normalize(x, 1)


class prediction_MLP(nn.Module):
    def __init__(
        self, in_dim=2048, hidden_dim=512, out_dim=2048
    ):  # bottleneck structure
        super().__init__()
        '''
        Prediction MLP. The prediction MLP (h) has BN applied
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers.
        The dimension of h’s input and output (z and p) is d = 2048,
        and h’s hidden layer’s dimension is 512, making h a
        bottleneck structure (ablation in supplement).
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
