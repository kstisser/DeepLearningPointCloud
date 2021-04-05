from DataTools import defs
from sklearn.cluster import DBSCAN
from . import clusterProfiler
from Visualization import visualizer as vis
import numpy as np

class Clusterer:
    def __init__(self, clusterType, data, eps = 0.05, minSamples=10):
        self.clusterType = clusterType
        self.eps = eps
        self.minSamples = minSamples
        self.data = data
        self.clusterThreshold = 0.7
        self.viz = vis.Visualizer()

    def determineClusteringParameters(self):
        #TODO- add statistical method to determine what the clustering parameters should be
        pass

    def cluster(self):
        if self.clusterType == defs.ClusterType.SKDBSCAN:
            self.dbscanSKLearn()
        elif self.clusterType == defs.ClusterType.O3DDBSCAN:
            self.dbscanO3D()
        else:
            print("Error! Don't recognize clustering type, won't do it!!!")

    def dbscanSKLearn(self):
        self.dbs = DBSCAN(eps=self.eps, min_samples=self.minSamples)
        self.dbs_labels = self.dbs.fit_predict(self.data)
        self.uniqueClusterLabels = set(self.dbs.labels_)
        print(self.dbs)
        print(self.uniqueClusterLabels)
        print("Found ", len(self.uniqueClusterLabels), " clusters!")
        self.clusters = self.dbs
        self.viz.visualizeClusters(self.data, self.dbs_labels)

    def dbscanO3D(self):
        self.dbs03d = self.data.cluster_dbscan(self.eps, self.minSamples, print_progress=True)
        print("DBSCAN O3D Results:")
        print(self.dbs03d)
        print(self.sbs03d.shape)
        self.clusters = self.dbs03d

    #This function works with the cluster profiler to compare each cluster against 
    #our expected face cluster attributes. If it is too far away, we won't keep it
    def getClustersWorthKeeping(self):
        cp = clusterProfiler.ClusterProfiler()
        self.keptClusters = []

        for i in self.uniqueClusterLabels:
            #check all real clusters, don't do anything with noise (-1)
            if (i != -1):
                cluster = self.data[np.where(self.dbs_labels == i)]
                clusterScore = cp.scoreCluster(cluster, self.clusterThreshold)
                if (clusterScore < self.clusterThreshold) and (clusterScore != -1):
                    print("Keeping cluster label: ", i, " with a score: ", clusterScore)
                    self.keptClusters.append(cluster)
        return self.keptClusters