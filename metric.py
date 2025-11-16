import numpy as np

def bagged_score(accuracy_scores):
    return np.mean([a["P"] for a in accuracy_scores])