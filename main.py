from cluster.kmeans import KmeansClusterTeo


if __name__ == "__main__":
    km = KMeansClusterTeo(4, 666)
    # labels = km.get_clustering_features("/project/graziul/ra/team_ser/data/zone1-20180812.wav", 1)
    labels = km.get_clustering_features("/project/graziul/ra/team_ser/data/new-201808120005-475816-27730.wav", 1)
    counts = km.aggregate(40)
    score = km.evaluate()
    print(f"Clustering score: {score}")
