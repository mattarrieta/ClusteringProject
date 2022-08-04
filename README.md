This is a package to cluster groups of data fast.

The main function can be used to see a step by step process of how the algorithm works and all the parameters it takes. The path of a folder of files can be used to separate the documents into a distinguishable files. Additionally, if the files are already split into sub categories, the performance of clustering can be evaluated.

The text is first extracted and processed using the getText file where puncutation, URLs, stopwords and other things are removed and the words can be stemmed or lemmatized.

The data is then vectorized in the DataManipulation file with TFIDF by default, but other ways can be adjusted. The data is then reduced by using various dimension reduction algorithms.

ClusterData is used to group the vectorizations (usually with HDBSCAN) and the predictions of the category of each document is given. The conservativeness of clustering can be adjusted but there is a default suggested. Evaluation metrics such as homogenitiy, and completeness can be used to evaluate the clustering if the labels are known.

Keywords are extracted from the predicicted groupings with several keyword extraction options.

The files groups can then be saved on to a directory.

Visualization of the model can be done with plotResults but a umap dimension reducation is recommended and may take more time than other dimension reductions. Several different plot types can be chosen from the libarary.

__Package Versions:__ hdbscan 0.8.28, keybert 0.5.1, pandas 1.4.2, nltk 3.7, numpy 1.21.5,regex 2022.3.15, scikit-learn 1.02

__Optional Package Versions:__ gensim 4.1.2, sentence_transformers 2.2.2, scikit-fuzzy 0.4.2, summa 1.2.0, yake 0.4.8, matplotlib 3.5.1, seaborn 0.11.2, umap-learn 0.5.3