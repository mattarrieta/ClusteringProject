import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from collections import OrderedDict
import pandas as pd



def plotResults(plotType, clusterer, pred_labels, reducedVector):
    """Plot results with given plotType. 

        Parameters:
            plotType(string): Option to plot data from "MST", "single linkage", "condensed Tree", or "vectorize". Must use HDBSCAN unless
                              using "vectorize". 
            clusterer(variable): Clusterer used to sort data. The clusterer must be an HDBSCAN in order to produce the Minimum Spanning Tree,
                                 single linkage tree, or condensed tree.
            pred_labels(list[string]): Predicted labels for each data according to reducedVector
            reducedVector(pandas DataFrame): Reduced vectorization of data to plot.
    """
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(10)
    if(plotType == "MST"):
        clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=20, edge_linewidth=2)
        plt.title("Minimum Spanning Tree")
        plt.show()
    if(plotType == "single linkage"):
        clusterer.single_linkage_tree_.plot()
        plt.title("Single Linkage Tree Plot")
        plt.show()
    if(plotType == "condensed tree"):
        clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette('deep'))
        plt.title("Condensed Tree Plot")
        plt.show()
    if(plotType == "vectorize"):
        reducedVectorArray = reducedVector.to_numpy()
        #Get colors for points to plot
        clusterProbabilities = clusterer.probabilities_
        only_pos = [num for num in list(set(pred_labels)) if num >= 0]
        color_palette = sns.color_palette("pastel", n_colors = len(only_pos))
        cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in pred_labels]
        #Adjust color saturation based on density
        cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterProbabilities)]
        #Organize colors for centroids
        cluster_colors_2 = []
        color_dict = {}
        for x in pred_labels:
            if x >= 0:
                cluster_colors_2.append(color_palette[x])
                color_dict[x] = color_palette[x]
            else:
                color_dict[x] = (0.5, 0.5, 0.5)
                cluster_colors_2.append((0.5, 0.5, 0.5))
        color_dict = OrderedDict(sorted(color_dict.items()))
        ListColors = (list(color_dict.items()))
        colorByList = []
        for x in ListColors:
            colorByList.append(x[1])
        #Use mean to find centroids
        colorDF = pd.DataFrame(data = {"Cluster Labels": pred_labels, "Colors": cluster_member_colors, "X1": reducedVectorArray[:,0], "X2": reducedVectorArray[:,1]})
        clusterGroups = colorDF.groupby(by = "Cluster Labels").mean()
        centroidX1 = clusterGroups["X1"]
        centroidX2 = clusterGroups["X2"]
        clusterGroups["Color"] = colorByList
        #Plot vectorizations and centroids
        plt.scatter(reducedVectorArray[:,0], reducedVectorArray[:,1], s=25, c=cluster_member_colors, alpha=1)
        plt.scatter(centroidX1, centroidX2, s = 20, c = 'black', marker = "v")
        plt.title("Vectorization Plot")
        plt.show()
    if(plotType == "vectorize hull"):
        reducedVectorArray = reducedVector.to_numpy()
        #Get colors for points to plot
        clusterProbabilities = clusterer.probabilities_
        only_pos = [num for num in list(set(pred_labels)) if num >= 0]
        color_palette = sns.color_palette("pastel", n_colors = len(only_pos))
        cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in pred_labels]
        #Adjust color saturation based on density
        cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterProbabilities)]
        #Organize colors for centroids
        cluster_colors_2 = []
        color_dict = {}
        for x in pred_labels:
            if x >= 0:
                cluster_colors_2.append(color_palette[x])
                color_dict[x] = color_palette[x]
            else:
                color_dict[x] = (0.5, 0.5, 0.5)
                cluster_colors_2.append((0.5, 0.5, 0.5))
        color_dict = OrderedDict(sorted(color_dict.items()))
        ListColors = (list(color_dict.items()))
        colorByList = []
        for x in ListColors:
            colorByList.append(x[1])
        #Use mean to find centroids
        colorDF = pd.DataFrame(data = {"Cluster Labels": pred_labels, "Colors": cluster_member_colors, "X1": reducedVectorArray[:,0], "X2": reducedVectorArray[:,1]})
        clusterGroups = colorDF.groupby(by = "Cluster Labels").mean()
        centroidX1 = clusterGroups["X1"]
        centroidX2 = clusterGroups["X2"]
        clusterGroups["Color"] = colorByList
        X1pointsByCat = colorDF.groupby("Cluster Labels")["X1"].apply(list)
        X2pointsByCat = colorDF.groupby("Cluster Labels")["X2"].apply(list)
        X1pointsByCatList = list(X1pointsByCat)
        X2pointsByCatList = list(X2pointsByCat)
        #Organize points by category and find outer points of polygon containing cluster
        for i in range(len(X1pointsByCatList)):
            curPoints = []
            if(len(X1pointsByCatList[i]) <= 2):
                continue
            for x in range(len(X1pointsByCatList[i])):
                curPoints.append([X1pointsByCatList[i][x], X2pointsByCatList[i][x]])
            npPoints = np.array(curPoints)
            hull = ConvexHull(npPoints)
            x_hull = np.append(npPoints[hull.vertices,0], npPoints[hull.vertices,0][0])
            y_hull = np.append(npPoints[hull.vertices,1], npPoints[hull.vertices,1][0])
            # plot shape
            
            plt.fill(x_hull, y_hull)
        plt.title("Vectorization Hull Plot")
        plt.show()
    else:
        return
