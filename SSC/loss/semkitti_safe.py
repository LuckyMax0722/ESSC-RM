import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

semantic_kitti_class_frequencies = np.array(
    [
        5.41773033e09,
        1.57835390e07,
        1.25136000e05,
        1.18809000e05,
        6.46799000e05,
        8.21951000e05,
        2.62978000e05,
        2.83696000e05,
        2.04750000e05,
        6.16887030e07,
        4.50296100e06,
        4.48836500e07,
        2.26992300e06,
        5.68402180e07,
        1.57196520e07,
        1.58442623e08,
        2.06162300e06,
        3.69705220e07,
        1.15198800e06,
        3.34146000e05,
    ]
)

kitti_class_names = [
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]


def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term

def safe_softmax(logits, dim=1):
    eps = 1e-5
    probs = F.softmax(logits, dim=dim)
    return probs.clamp(min=eps, max=1.0-eps)
    
def geo_scal_loss(pred, ssc_target, ignore_index=255, non_empty_idx=0):

    # Get softmax probabilities
    pred = safe_softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, non_empty_idx]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != ignore_index
    nonempty_target = ssc_target != non_empty_idx
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum()+eps)
    recall = intersection / (nonempty_target.sum()+eps)
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / ((1 - nonempty_target).sum()+eps)
    
    precision = precision.clamp(min=eps, max=1.0-eps)
    recall  = recall.clamp(min=eps, max=1.0-eps)
    spec = spec.clamp(min=eps, max=1.0-eps)

    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss(pred, ssc_target, ignore_index=255):
    # Get softmax probabilities
    pred = safe_softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != ignore_index
    n_classes = pred.shape[1]

    eps = 1e-5

    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:

                precision = nominator / (torch.sum(p) + eps)
                precision = precision.clamp(min=eps, max=1.0 - eps)

                loss_precision = F.binary_cross_entropy(precision, torch.ones_like(precision))
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:

                recall = nominator / (torch.sum(completion_target) + eps)
                recall = recall.clamp(min=eps, max=1.0 - eps)

                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:

                specificity = torch.sum((1 - p) * (1 - completion_target)) / (torch.sum(1 - completion_target)  + eps)
                specificity = specificity.clamp(min=eps, max=1.0 - eps)

                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class

    return loss / (count + eps)


def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss

def vel_loss(pred, gt):
    return F.l1_loss(pred, gt)


def KL_loss(pna_branch, text_branch):
    """
    KL Loss
    """
    prob1 = F.softmax(pna_branch, dim=1)
    prob2 = F.softmax(text_branch, dim=1)
    
    log_prob1 = F.log_softmax(pna_branch, dim=1)
    log_prob2 = F.log_softmax(text_branch, dim=1)
    
    kl_1_to_2 = F.kl_div(log_prob1, prob2)
    kl_2_to_1 = F.kl_div(log_prob2, prob1)
    
    symmetric_kl = (kl_1_to_2 + kl_2_to_1) / 2

    return symmetric_kl
