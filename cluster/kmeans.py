from sklearn.cluster import KMeans
import sys
sys.path.append("..")
from feature.teo_cb_auto_env import TeagorExtractor
import numpy as np
from typing import List

from cluster_evaluator import ClusterEvaluator


class KMeansClusterTeo:
    def __init__(self, k, random_state):
        self.k = k
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters = self.k, random_state = self.random_state)
        self.interval = 0
        self.unit = 0
        self.teo = TeagorExtractor()
        self.features = None
        self.labels = None
        self.evaluator = ClusterEvaluator()

    def cluster_in_day(
            self, 
            zone: str,
            year: str, 
            month: str, 
            date: str
        ):
        assert len(month) == 2, "Please make sure the argument month is like 'hh'!"
        assert len(date) == 2, "Please make sure the argument date is like 'mm'!"

        pass

    def get_clustering_features(self, filepath, delta):
        # teo = TeagorExtractor()
        self.interval = delta
        teo_features: np.array = self.teo.process(filepath, delta)
        print(f"Input features size: {teo_features.shape}")
        # print(teo_features)
        self.features = np.nan_to_num(teo_features)
        self.labels = self.kmeans.fit_predict(self.features)
        print(f"Output labels size: {self.labels.shape}")
        return self.labels

    def aggregate(self, unit: int) -> List[List]:
        self.unit = unit
        all_counts = []
        unit_counts = [0 for i in range(self.k)]
        accum = 0.
        for i in range(len(self.labels)):
            unit_counts[self.labels[i]] += 1
            accum += self.interval
            if accum % self.unit == 0:
                all_counts.append(unit_counts)
                unit_counts = [0 for i in range(self.k)]
        
        all_counts.append(unit_counts)
        all_counts = np.array(all_counts)
        print(f"Aggregated feature size: {all_counts.shape}")
        return all_counts

    def evaluate(self):
        self.score = self.evaluator.dbi(self.features, self.labels)
        return self.score


if __name__ == "__main__":
    km = KMeansClusterTeo(4, 666)
    # labels = km.get_clustering_features("/project/graziul/ra/team_ser/data/zone1-20180812.wav", 1)
    labels = km.get_clustering_features("/project/graziul/ra/team_ser/data/new-201808120005-475816-27730.wav", 1)
    counts = km.aggregate(40)
    score = km.evaluate()
    print(f"Clustering score: {score}")
    np.savetxt("counts.txt", counts)
