import pandas as pd
import os
import numpy as np
import time


class StandardVectorization:
    """ Class of common vectorization methods.
        :param data :type list[str]: list of text documents
        :keyword labels :type Optional list[str]: list of corresponding document names to data text documents  """
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels if labels is not None else range(len(self.data))
        self.unique_labels = set(self.labels)

    def __method_vectorize(self, vectorizer, data: str) -> pd.DataFrame:
        """ :return a dataframe with the data transformed by the vecotrizer"""
        fitted_data = vectorizer.fit_transform(data)
        tokens = vectorizer.get_feature_names_out()
        dataframe = pd.DataFrame(data=fitted_data.toarray(), index=self.labels, columns=tokens)
        return dataframe

    def count(self, binary=False, min_df=0) -> pd.DataFrame:
        """ Create count vectorizer.
            :return A pd.DataFrame of vectorized text with token columns.  """
        from sklearn.feature_extraction.text import CountVectorizer
        return self.__method_vectorize(CountVectorizer(stop_words='english', binary=binary, min_df=min_df), self.data)

    def tfidf(self, min_df=0.0, ngrams=(1, 1), max_df=1.0, max_features=None) -> pd.DataFrame:
        """ Create TFIDF vectorizer
            :return A pd.DataFrame of vectorized text with token columns.  """
        from sklearn.feature_extraction.text import TfidfVectorizer
        return self.__method_vectorize(TfidfVectorizer(stop_words='english', min_df=min_df, max_features=max_features,
                                                       ngram_range=ngrams, max_df=max_df),
                                       self.data)

    def doc2vec(self) -> pd.DataFrame:
        """ Vectorize using Doc2Vec
            :return a pd.DataFrame of vectorized text with token columns."""
        from gensim.models.doc2vec import Doc2Vec
        try:
            model = Doc2Vec.load(rf'{os.getcwd()}\doc2vec-train-model')
        except FileNotFoundError:
            from __TrainFunctions__ import doc2vec_train
            model = doc2vec_train()
        word_lists = [self.data[i].split() for i in range(len(self.data))]
        vectors = [model.infer_vector(word_lists[i]) for i in range(len(word_lists))]
        return pd.DataFrame(vectors)

    def bert(self, model_type='all-mpnet-base-v2') -> pd.DataFrame:
        """ Vectorize using BERT
            :return A pd.DataFrame of vectorized text with token columns.  """
        try:
            return pd.read_csv(rf'{os.getcwd()}\BertModels\{model_type}')
        except FileNotFoundError:
            from __TrainFunctions__ import bert_train
            return bert_train(model_type, self.data, labels=self.labels)


class AdjustedVectorization(StandardVectorization):
    """ Augmented vectorization methods on StandardVectorization methods.  """
    def __init__(self, data, labels=None):
        StandardVectorization.__init__(self, data, labels)

    def __adjusted_vectorization(self, min_freq: float, min_df: int) -> list[str]:
        """ :param min_freq: if the tfidf score of a token is below min_freq for all files, remove token
            :param min_df: the minimum number or frequency that a token needs to appear in the dataset
            NOTE: min_freq is for relevancy of token for each file, min_df is for relevancy of token across dataset  """
        tfidf_df = self.tfidf(min_df=min_df)
        irrelevant_words = []
        for word in tfidf_df.columns:
            if not tfidf_df[word].ge(min_freq).any():
                irrelevant_words.append(word)
        return irrelevant_words

    def count_min_freq(self, min_freq=0.1, min_df=0) -> pd.DataFrame:
        """ Remove CountVectorizer tokens if TfidfVectorizer score is below min_freq.
            :return A pandas.DataFrame of vectorized text with token columns.  """
        irrelevant_words = self.__adjusted_vectorization(min_freq, min_df=min_df)
        count_df = self.count(min_df=min_df)
        return count_df.drop(columns=irrelevant_words)

    def tfidf_min_freq(self, min_freq=0.1, min_df=0) -> pd.DataFrame:
        """ Remove TfidfVectorizer tokens if TfidfVectorizer score is below min_freq.
            :return A pandas.DataFrame of vectorized text with token columns.  """
        irrelevant_words = self.__adjusted_vectorization(min_freq, min_df=min_df)
        method_df = self.tfidf(min_df=min_df)
        return method_df.drop(columns=irrelevant_words)

    def tfidf_weighted(self, min_df=0, max_df=1.0, max_features=None) -> pd.DataFrame:
        """ Weight TfidfVectorizer ngrams=2 tokens to be half of ngrams=1 tokens.
            :return A pandas.DataFrame of vectorized text with token columns. """
        df1 = self.tfidf(min_df=min_df, max_df=max_df, ngrams=(1, 1), max_features=max_features)
        df2 = self.tfidf(min_df=min_df, max_df=max_df, ngrams=(2, 2), max_features=max_features)
        df2 = df2.multiply(.5)
        return pd.concat([df1, df2], axis=1)


class DimensionReduction:

    """ Class of common dimension reduction methods on pandas.DataFrames
        NOTE: For param value storage. Reference use only. Class does not allow for param changes outside of method_dict
        reduced_df :type pd.DataFrame: dimension-reduced dataframe to given dim using given reduction_model

        Parameters:
            df(pandas DataFrame): Vectorization of data
            reduction_model(string): Type of reduction to be done from isomap', 'pca-standard', 'pca-incremental', 
                                     'pca-kernel', 'svd', 'tsne' or 'umap'.
            dim(int): Number of dimensions to reduce to
            n_neighbors(int) = Balances between maintaining local and globabl stucture 
            min_dist(int) = How tightle points are clumped together
        Variables:
            self.reduced_df(pandas DataFrame): DataFrame with reduced vectiorization;

    """
    def __init__(self, df, reduction_model='svd', dim=20, n_neighbors = 10, min_dist = 0.1):
        from sklearn.manifold import Isomap
        from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
        from sklearn.manifold import TSNE
        from sklearn.decomposition import TruncatedSVD

        #UMAP takes a long time to import so we don't import it unless it's needed
        if(reduction_model == 'umap'):
            import umap
            model_dict = {
                'isomap': Isomap(n_components=dim, n_neighbors=10, eigen_solver='auto', tol=0, max_iter=None,
                                path_method='auto', neighbors_algorithm='auto', n_jobs=None, metric='cosine', p=2,
                                metric_params=None),
                'pca-standard': PCA(n_components=dim, copy=True, whiten=False,
                                    svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None),
                'pca-incremental': IncrementalPCA(n_components=dim, whiten=False, copy=True, batch_size=None),
                'pca-kernel': KernelPCA(n_components=dim, kernel='linear', gamma=None, degree=3, coef0=1,
                                        kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver='auto',
                                        tol=0, max_iter=None, iterated_power='auto', remove_zero_eig=False,
                                        random_state=None, copy_X=True, n_jobs=None),
                'svd': TruncatedSVD(n_components=dim, algorithm='randomized', n_iter=5, random_state=None, tol=0.0),
                'tsne': TSNE(n_components=dim, perplexity=30, early_exaggeration=12.0, learning_rate=200.0,
                            n_iter=1000, min_grad_norm=1e-7, metric='cosine', init='pca', verbose=0,
                            method='exact', angle=.5, n_jobs=-1),
                'umap': umap.UMAP(metric = 'cosine', n_neighbors = n_neighbors , min_dist = min_dist)
            }
        else:
            model_dict = {
                'isomap': Isomap(n_components=dim, n_neighbors=10, eigen_solver='auto', tol=0, max_iter=None,
                                path_method='auto', neighbors_algorithm='auto', n_jobs=None, metric='cosine', p=2,
                                metric_params=None),
                'pca-standard': PCA(n_components=dim, copy=True, whiten=False,
                                    svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None),
                'pca-incremental': IncrementalPCA(n_components=dim, whiten=False, copy=True, batch_size=None),
                'pca-kernel': KernelPCA(n_components=dim, kernel='linear', gamma=None, degree=3, coef0=1,
                                        kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver='auto',
                                        tol=0, max_iter=None, iterated_power='auto', remove_zero_eig=False,
                                        random_state=None, copy_X=True, n_jobs=None),
                'svd': TruncatedSVD(n_components=dim, algorithm='randomized', n_iter=5, random_state=None, tol=0.0),
                'tsne': TSNE(n_components=dim, perplexity=30, early_exaggeration=12.0, learning_rate=200.0,
                            n_iter=1000, min_grad_norm=1e-7, metric='cosine', init='pca', verbose=0,
                            method='exact', angle=.5, n_jobs=-1),
            }

        self.reduced_df = pd.DataFrame(model_dict[reduction_model].fit_transform(df), index=df.index)



def reduceVector(vectorizedData, reduction_method = 'svd', n_neighbors = 10, min_dist = 0.1):
    """Helper function to reduce the dimensions of the given vecotrized data with the chosen reduction method and paramerters"""

    reducedData = DimensionReduction(vectorizedData, reduction_model= reduction_method, n_neighbors= n_neighbors, min_dist = min_dist).reduced_df
    
    return reducedData
