import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import random
import tensorflow as tf
from collections import defaultdict
import os

def cosine_similarity(features, threshold=None):
    # Feature normalization (convert each row to a unit vector)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_norm = features / np.maximum(norms, 1e-10)  # Avoid division by zero

    # Calculate cosine similarity (dot product of feature matrix and its transpose)
    similarity = np.dot(features_norm, features_norm.T)

    # Threshold filtering (optional)
    if threshold is not None:
        similarity[similarity < threshold] = 0.0

    return similarity


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


def normalize_features(feat):
    degree = np.asarray(feat.sum(1)).flatten()

    # Set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(train_arr, test_arr):
    cache_path = r"/HMDAD/cache.npz"

    # Check if the cache exists
    if os.path.exists(cache_path):
        print("Loading precomputed feature cache...")
        cache = np.load(cache_path)
        interaction = cache['interaction']
        features = cache['features']
        # Load the label file
        labels = np.loadtxt(r"HMDAD/adj.txt")

        # Calculate nd and nm from the labels
        nd = np.max(labels[:, 0]).astype(np.int32)
        nm = np.max(labels[:, 1]).astype(np.int32)
    else:
        print("First run, computing features and caching...")
        labels = np.loadtxt(r"HMDAD/adj.txt")

        nd = np.max(labels[:, 0]).astype(np.int32)
        nm = np.max(labels[:, 1]).astype(np.int32)

        # Load the disease-microbe interaction matrix
        M = sio.loadmat(r"HMDAD/interaction.mat")
        M = M['interaction']

        F1 = np.loadtxt(r"HMDAD/microbe_features.txt")
        F2 = np.loadtxt(r"HMDAD/disease_features.txt")

        # Calculate the feature similarity matrix of homogeneous nodes
        disease_similarity = cosine_similarity(F1, threshold=0.5)
        microbe_similarity = cosine_similarity(F2, threshold=0.5)

        # Construct the heterogeneous network adjacency matrix
        interaction = np.vstack((
            np.hstack((disease_similarity, M)),
            np.hstack((M.transpose(), microbe_similarity))
        ))

        # Construct the feature matrix
        features = np.vstack((
            np.hstack((F1, np.zeros(shape=(F1.shape[0], F2.shape[1]), dtype=int))),
            np.hstack((np.zeros(shape=(F2.shape[0], F1.shape[0]), dtype=int), F2))
        ))
        features = normalize_features(features)

        # Save the cache to the specific file
        np.savez(cache_path, interaction=interaction, features=features)
        print(f"Features cached to {cache_path}")

    # Process test set indices to ensure non-negativity
    test_row = labels[test_arr, 0] - 1
    test_col = labels[test_arr, 1] - 1
    test_row = np.maximum(test_row, 0)
    test_col = np.maximum(test_col, 0)
    logits_test = sp.csr_matrix((labels[test_arr, 2], (test_row, test_col)),
                                shape=(nd, nm)).toarray()
    logits_test = logits_test.reshape([-1, 1])

    # Process training set indices to ensure non-negativity
    train_row = labels[train_arr, 0] - 1
    train_col = labels[train_arr, 1] - 1
    train_row = np.maximum(train_row, 0)
    train_col = np.maximum(train_col, 0)
    logits_train = sp.csr_matrix((labels[train_arr, 2], (train_row, train_col)),
                                 shape=(nd, nm)).toarray()
    logits_train = logits_train.reshape([-1, 1])

    train_mask = np.array(logits_train[:, 0], dtype=bool).reshape([-1, 1])
    test_mask = np.array(logits_test[:, 0], dtype=bool).reshape([-1, 1])

    return interaction, features, sparse_matrix(logits_train), logits_test, train_mask, test_mask, labels


def generate_mask(labels, N):
    num = 0

    nd = np.max(labels[:, 0])
    nm = np.max(labels[:, 1])
    nd = nd.astype(np.int32)
    nm = nm.astype(np.int32)

    A = sp.csr_matrix((labels[:, 2], (labels[:, 0] - 1, labels[:, 1] - 1)), shape=(nd, nm)).toarray()
    mask = np.zeros(A.shape)
    label_neg = np.zeros((1 * N, 2))
    while (num < 1 * N):
        a = random.randint(0, nd - 1)
        b = random.randint(0, nm - 1)
        if A[a, b] != 1 and mask[a, b] != 1:
            mask[a, b] = 1
            label_neg[num, 0] = a
            label_neg[num, 1] = b
            num += 1
    mask = np.reshape(mask, [-1, 1])
    return mask, label_neg


def test_negative_sample(labels, N, negative_mask):
    num = 0
    (nd, nm) = negative_mask.shape
    A = sp.csr_matrix((labels[:, 2], (labels[:, 0] - 1, labels[:, 1] - 1)), shape=(nd, nm)).toarray()
    mask = np.zeros(A.shape)
    test_neg = np.zeros((1 * N, 2))
    while (num < 1 * N):
        a = random.randint(0, nd - 1)
        b = random.randint(0, nm - 1)
        if A[a, b] != 1 and mask[a, b] != 1:
            mask[a, b] = 1
            test_neg[num, 0] = a
            test_neg[num, 1] = b
            num += 1
    return test_neg


def div_list(ls, n):
    ls_len = len(ls)
    j = ls_len // n
    ls_return = []
    for i in range(0, (n - 1) * j, j):
        ls_return.append(ls[i:i + j])
    ls_return.append(ls[(n - 1) * j:])
    return ls_return


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def maxpooling(a):
    a = tf.cast(a, dtype=tf.float32)
    b = tf.reduce_max(a, axis=1, keepdims=True)
    c = tf.equal(a, b)
    mask = tf.cast(c, dtype=tf.float32)
    final = tf.multiply(a, mask)
    ones = tf.ones_like(a)
    zeros = tf.zeros_like(a)
    final = tf.where(final > 0.0, ones, zeros)
    return final


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized


def sparse_matrix(matrix):
    sigma = 0.001
    matrix = matrix.astype(np.int32)
    result = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        if matrix[i, 0] == 0:
            result[i, 0] = sigma
        else:
            result[i, 0] = 1
    return result