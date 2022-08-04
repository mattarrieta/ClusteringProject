import time

from regex import D
from DataManipulation import AdjustedVectorization, reduceVector, fastVectorize
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
        start_time_text = time.perf_counter()

        self.data, self.labels, self.filenames = getText(data_folder_path, knowLabels = know_labels, preprocessData = True, RemoveNums = True, lem=True, extendStopWords= True)
        end_time_text = time.perf_counter()
        print("Text Processing:", end_time_text - start_time_text)
        #print(self.data[0])
        #Vectorize data
        start_time_vectorize = time.perf_counter()
        print("Vectorize")
        #self.vectorized = AdjustedVectorization(self.data, labels=self.labels).tfidf()
        self.vectorized = fastVectorize(self.data, self.labels)
        #print(len(self.vectorized))
        #print(len(self.vectorized.columns))

        #print(myVectorize)
        end_time_vectorize = time.perf_counter()
        print("Vectorize Time:", end_time_vectorize - start_time_vectorize)


        #Reduce the vectorized data
        start_time_reduce = time.perf_counter()
        print("Reduction")
        self.reducedDF = reduceVector(self.vectorized, reduction_method = reduction_method)
        end_time_reduce = time.perf_counter()
        print("Reduction Time:", end_time_reduce - start_time_reduce)

        #Cluster data

        start_time_cluster = time.perf_counter()
        print("Cluster")
        self.cluster = clusterData(self.reducedDF, setting = setting)


        #Get predicted groups of each file
        self.pred_labels = getPredicitonLabels(self.cluster)
        end_time_cluster = time.perf_counter()
        print("Cluster Time:", end_time_cluster - start_time_cluster)


        #If the known labels are given calculate accuracy metrics
        start_time_metrics = time.perf_counter()
        print("Metrics")
        if(self.labels != None):
            self.homogenity , self.completenss, self.nmi = getScores(self.labels, self.pred_labels)
        end_time_metrics = time.perf_counter()
        print("Metric Time:", end_time_metrics - start_time_metrics)

        #Get the keywords for each data file
        start_time_keyword = time.perf_counter()
        #print("Keywords")
        self.keywords, self.pred_labels_dict = assign_keywords(self.pred_labels, self.data)
        end_time_keyword = time.perf_counter()
        print("Keyword Time:", end_time_keyword - start_time_keyword)

        if(printResults):
            predDF = pd.DataFrame(self.pred_labels_dict, index = self.labels)
            print(predDF.to_string())

        #Save predicted documents into folders if needed
        start_time_save = time.perf_counter()
        #print("Save")
        if save is not None:
            save_folders(save, "TestClusters", self.pred_labels_dict, self.filenames, self.data)
        end_time_save = time.perf_counter()
        print("Save Time:", end_time_save - start_time_save)

        #plot the results if needed
        start_time_plot = time.perf_counter()
        print("Save")
        plotResults(plotType = plotType, clusterer=self.cluster.clusterer, pred_labels = self.pred_labels, reducedVector=self.reducedDF)
        end_time_plot = time.perf_counter()
        print("Plot Time:", end_time_plot - start_time_plot)
        print("Time no text processing:", end_time_plot - start_time_vectorize)

def main(file_path):  # TODO: Check labels!!!
    start = time.perf_counter()
    clus = ClusterDataFast(file_path, save=0, setting = 0, printResults=False, reduction_method="svd")
    print(clus.homogenity)
    print(clus.completenss)
    end = time.perf_counter()
    print('time', end - start)


if __name__ == '__main__':
    main(r"C:\Users\Matthew Arrieta\Desktop\FinalClusteringProject\Test Set Outliers Grouped")