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

    def metric(self, prediction, reference, m):
        P = prediction.split("\n\n")
        R = reference.split("\n\n")
        C = np.zeros(shape=(len(R), len(P)))

        for i, r in enumerate(R):
            for j, p in enumerate(P):
                C[i][j] = m(p, r)[0]
        
        C = self.squarify(C)
        C_copy = copy.deepcopy(C)

        mnkrs = Munkres()
        best_idxs = mnkrs.compute(C)

        t = np.sum([C_copy[r][p] for r, p in best_idxs])

        precision = t / len(P)
        recall = t / len(R)
        F1 = 2 * precision * recall / (precision + recall)
        
        return precision, recall, F1