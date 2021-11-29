from sklearn.cluster import DBSCAN

from kmeans import ClusterTeo


class DBSCANClusterTeo(ClusterTeo):
	def __init__(self, eps, min_samples):
		ClusterTeo.__init__(self)
		self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
		
	def get_clustering_labels(self, filepath, delta):
		self.get_clustering_features(filepath, delta)
		self.labels = self.dbscan.fit_predict(self.features)
		print(f"Output labels size: {self.labels.shape}")
		return self.labels


if __name__ == "__main__":
    eps, min_samples = 0.00000000001, 100
    db = DBSCANClusterTeo(eps, min_samples)
    print(f"DBSCAN Clustering eps: {eps}, min_samples: {min_samples}")
    # labels = db.get_clustering_labels("/project/graziul/ra/team_ser/data/zone1-20180812.wav", 1)
    print(f"Clustering interval: 2")
    labels = db.get_clustering_labels("/project/graziul/ra/team_ser/data/new-201808120005-475816-27730.wav", 2)
    counts = db.aggregate(40)
    score = db.evaluate()
    print(f"Clustering score: {score}")
    np.savetxt("counts_db_0.txt", counts)

