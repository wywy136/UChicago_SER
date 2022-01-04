from sklearn.cluster import KMeans
import sys
sys.path.append("..")
from feature.teo_cb_auto_env import TeagorExtractor
import numpy as np
from typing import List

from cluster_evaluator import ClusterEvaluator


class ClusterTeo:
	def __init__(self):
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

	def aggregate(self, unit: int) -> List[List]:
		self.unit = unit
		all_counts = []
		label_size = self.get_label_size(self.labels)
		print(f"Num of clusters: {label_size}")
		unit_counts = [0 for i in range(label_size)]
		accum = 0.
		for i in range(len(self.labels)):
			unit_counts[self.labels[i]] += 1
			accum += self.interval
			if accum % self.unit == 0:
				all_counts.append(unit_counts)
				unit_counts = [0 for i in range(label_size)]
        
		all_counts.append(unit_counts)
		all_counts = np.array(all_counts)
		print(f"Aggregated feature size: {all_counts.shape}")
		return all_counts

	def evaluate(self):
		self.score = self.evaluator.dbi(self.features, self.labels)
		return self.score

	def get_label_size(self, labels):
		disparate = []
		for n in labels:
			if n not in disparate:
				disparate.append(n)

		return len(disparate)

	def get_clustering_features(self, filepath: str, delta: int):
		self.interval = delta
		teo_features: np.array = self.teo.process(filepath, delta)
		print(f"Input features size: {teo_features.shape}")
        # print(teo_features)
		self.features = np.nan_to_num(teo_features)

	def get_clustering_labels(self, filepath: str, delta: int):
		raise NotImplementedError


class KMeansClusterTeo(ClusterTeo):
	def __init__(self, k, random_state):
	    ClusterTeo.__init__(self)
	    self.k = k
	    self.random_state = random_state
	    self.kmeans = KMeans(n_clusters = self.k, random_state = self.random_state)

	def get_clustering_labels(self, filepath, delta):
            self.get_clustering_features(filepath, delta)
            # print(f"Feature size: {self.features.shape}")
            self.labels = self.kmeans.fit_predict(self.features)
            print(f"Output labels size: {self.labels.shape}")
            return self.labels


if __name__ == "__main__":
    km = KMeansClusterTeo(2, 666)
    print(f"Kmeans Clustering number: 2")
    labels = km.get_clustering_labels("./zone1-20180814.wav", 2)
    print(f"Clustering interval: 2")
    #v labels = km.get_clustering_features("/project/graziul/ra/team_ser/data/new-201808120005-475816-27730.wav", 1)
    counts = km.aggregate(3600)
    score = km.evaluate()
    print(f"Clustering score: {score}")
    np.savetxt("counts_c2_i200_z1_20180814.txt", counts)

