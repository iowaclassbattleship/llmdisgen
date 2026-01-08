# https://arxiv.org/pdf/2405.01930

import numpy as np
from munkres import Munkres


def squarify(M, val=0):
    (a, b) = M.shape
    if a > b:
        padding = ((0, 0), (0, a - b))
    else:
        padding = ((0, b - a), (0, 0))
    return np.pad(M, padding, mode="constant", constant_values=val)


def split_string(s, chunk_size=512):
    """Split string s into chunks of size chunk_size."""
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


def metric(P, R, m, block_size=512):
    P = split_string(P, block_size)
    R = split_string(R, block_size)
    if len(P) == 0 or len(R) == 0:
        return 0, 0, 0
    C = np.zeros(shape=(len(R), len(P)))

    for i, r in enumerate(R):
        for j, p in enumerate(P):
            C[i][j] = m(p, r)[0]

    C_square = squarify(C)

    cost_matrix = -C_square

    mnkrs = Munkres()
    best_pairs = mnkrs.compute(cost_matrix.tolist())

    total_score = 0
    for r, p in best_pairs:
        if r < len(R) and p < len(P):
            total_score += C[r][p]

    precision = total_score / len(P)
    recall = total_score / len(R)

    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * precision * recall / (precision + recall)

    return precision, recall, F1
