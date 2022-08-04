import time
from pyspark import keyword_only

from regex import D
from DataManipulation import AdjustedVectorization, reduceVector
from getText import getText
from plotResults import plotResults
from clusteringMethods import clusterData, getPredicitonLabels, getScores
from extractKeywords import assign_keywords
from saveData import save_folders
import pandas as pd

class ClusterDataFast:
    """ Cluster similar text documents.

        Parameters:
            data_folder_path type (str):  path to folder containing text documents
            know_labels (bool): True if the categories are known and the documents are grouped by subfolders, False if all files are together
            save (string): if not None, saves text documents in folders at the given string if 0, then folders are saved in {cwd}/ClusteredData
            reduction_method (string): Reduction method choosen from 'isomap', 'pca-standard', 'pca-incremental', 'pca-kernel', 'svd', 'tsne',
                                       or 'umap'.
            setting (int): Option on how conservative clustering is. 0 is default but range is 1-5 with 1 having more smaller clusters and 5 
                           with less bigger clusters.
            plotType (string): Option to plot data from "MST", "single linkage", "condensed Tree", "vectorize", or "vectorize hull". Must use HDBSCAN unless
                            using "vectorize".
        Variables:
            self.data(list[str]): List of text files inputted
            self.labels(list[str]): List of labels known for the category of each file in data
            self.filenames(list[str]): List of filenames in order of data files
            self.vectorized(pandas DataFrame): Contains a matrix with the tfidf vectorization of the data
            self.reducedDF(pandas DataFrame): A matrix of the vectorized data after dimension reduction
            self.cluster(ClusteringMethod Object): A ClusteringMethod Object that clusters the reduced data
            self.pred_labels(list[int]): List of integers with the group number each document was clustered to in order of data
            self.homogenity(int): If labels are given the homogenity score, else not filled
            self.completenss(int): If labels are given the completeness score, else not filled
            self.nmi(int): If labels are given, the normalized mutual info score, else not filled
            self.keywords(dic{string: tuple(string)}): Labels with the keywords corresponding to each group
            self.pred_labels_dict(list[tuple(string: tuple(string))]): Labels and keywords for each document
    """
    def __init__(self, data_folder_path, know_labels = True, save=None, reduction_method = "svd", setting = 0, plotType = None, printResults = False):

        #Read in text information into respective variabels and preprocess if needed
        self.data, self.labels, self.filenames = getText(data_folder_path, knowLabels = know_labels, preprocessData = True, RemoveNums = True, lem=True, extendStopWords= True)
        
        #Vectorize data
        self.vectorized = AdjustedVectorization(self.data, labels=self.labels).tfidf(min_df = .005, max_df = .8, ngrams = (1,2))

        #Reduce the vectorized data
        self.reducedDF = reduceVector(self.vectorized, reduction_method = reduction_method)

        #Cluster data
        self.cluster = clusterData(self.reducedDF, setting = setting)

        #Get predicted groups of each file
        self.pred_labels = getPredicitonLabels(self.cluster)

        #If the known labels are given calculate accuracy metrics
        if(self.labels != None):
            self.homogenity , self.completenss, self.nmi = getScores(self.labels, self.pred_labels)

        #Get the keywords for each data file
        self.keywords, self.pred_labels_dict = assign_keywords(self.pred_labels, self.data)

        #Print results of how each file was categorized if wanted
        if(printResults):
            predDF = pd.DataFrame(self.pred_labels_dict, index = self.labels)
            print(predDF.to_string())

        #Save predicted documents into folders if needed
        if save is not None:
            save_folders(save, "TestClusters", self.pred_labels_dict, self.filenames, self.data)

        #plot the results if needed
        plotResults(plotType = plotType, clusterer=self.cluster.clusterer, pred_labels = self.pred_labels, reducedVector=self.reducedDF)


def main(file_path):  # TODO: Check labels!!!
    clus = ClusterDataFast(file_path, save=0, setting = 0, printResults=False, reduction_method="svd", plotType=None)
    print("Homogenity:", clus.homogenity)
    print("Completeness: ", clus.completenss)


if __name__ == '__main__':
    main(r"C:\Users\Matthew Arrieta\Desktop\FinalClusteringProject\Test Set Outliers Grouped")