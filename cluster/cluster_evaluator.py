from sklearn.metrics import davies_bouldin_score

from typing import List


class ClusterEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def dbi(X, labels):
        assert len(X) == len(labels)

        return davies_bouldin_score(X, labels)

    # TODO: more clustering evaluation metrics: silhouette, CH
