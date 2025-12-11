# https://arxiv.org/pdf/2405.01930

import numpy as np
from munkres import Munkres
import copy

class BlockMatch:
    def __init__(self, block_size=512):
        self.block_size = block_size

    def squarify(self, M, val=0):
        (a,b)=M.shape
        if a>b:
            padding=((0,0),(0,a-b))
        else:
            padding=((0,b-a),(0,0))
        return np.pad(M,padding,mode='constant',constant_values=val)

    def metric(self, P, R, m):
        C = np.zeros(shape=(len(R), len(P)))

        for i, r in enumerate(R):
            for j, p in enumerate(P):
                C[i][j] = m(p, r)[0]

        C_square = self.squarify(C)

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
