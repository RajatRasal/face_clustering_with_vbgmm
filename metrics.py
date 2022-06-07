# TODO: 16 images for each cluster group
# TODO: We want to be able to determine low prec or low rec clusters, and then associate images to those clusters
from collections import Counter

import scipy
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def metrics(true, pred):
    cluster_ids = np.unique(pred)
    pred_copy = np.empty_like(pred)
    for cluster_id in cluster_ids:
        # Mapping from predicted class ids to true class ids
        true_classes_in_cluster = true[pred == cluster_id]
        # Most occurring true class id is assigned to the cluster
        tp_class = Counter(true_classes_in_cluster).most_common(1)[0][0]
        # Mapping from predicted class ids to true class ids
        pred_copy[pred == cluster_id] = tp_class
    macro_precision = precision_score(true, pred_copy, average='macro')
    micro_precision = precision_score(true, pred_copy, average='micro')
    macro_recall = recall_score(true, pred_copy, average='macro')
    micro_recall = recall_score(true, pred_copy, average='micro')
    accuracy = accuracy_score(true, pred_copy)
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'micro_precision': micro_precision,
        'macro_recall': macro_recall,
        'micro_recall': micro_recall,
    }

def jensen_shannon_distance(true, pred, identities):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """
    true_dist = np.array([v for _, v in Counter(true).most_common(identities)])
    pred_dist = np.array([v for _, v in Counter(pred).most_common(identities)])
    # calculate m
    m = (true_dist + pred_dist) / 2
    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(true_dist, m) + scipy.stats.entropy(pred_dist, m)) / 2
    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)
    return distance
