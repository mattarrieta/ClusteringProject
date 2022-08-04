from collections import Counter
from DataManipulation import AdjustedVectorization
from groupData import GroupLabeledTextData

def assign_keywords(pred_labels, data, keyword_model = 'tfidf-cluster'):
    '''Extract keywords from predicted groups. keywords_dict has the keywords for each group, keywords has the
       each file's predicted cluster keywords associated with it.'''
    keywords_dict = ExtractKeywords((pred_labels,data), keyword_model=keyword_model).keywords_dict
    counter = Counter(pred_labels)
    keywords, num_outliers = [], 0
    for pred_lab in pred_labels:
        if counter[pred_lab] > 2:
            keywords.append((pred_lab, keywords_dict[pred_lab]))
        else:
            num_outliers += 1
            keywords.append(f'Outlier-{num_outliers}')
    return keywords_dict, keywords

class ExtractKeywords:
    """ Estimates the contents of a cluster of text data. If model is in {keybert, summa, tfidf, yake}, then keywords
        are extracted for every data point and most frequent keywords of each cluster is returned. If model is
        tfidf-cluster or LDA, then the keywords are directly extracted from a cluster (faster runtime).
        
        Parameters:
            param labeled_data(list[str], list[str]): Tuple with label of data and data itself.
            keyword_model(string): Model to extract keywords
            num_keywords(int): number of keywords to be returned for each cluster
            min_clust_size(int): the smallest number of grouped data points to be considered a cluster
        Variables:
            self.keywords_dict(dict{string: tuple(string)}): Dictionary with the keys being the labels and 
            values being the predicted keywords for each label
    """
    def __init__(self, labeled_data,  keyword_model='tfidf-cluster', num_keywords=3, min_clust_size=2):
        self.labels, self.data = labeled_data
        self.label_occurrences = Counter(self.labels)
        self.num_keywords, self.min_clust_size = num_keywords, min_clust_size
        self.keywords_dict = self.__get_keywords(keyword_model)

    @classmethod
    def __yake(cls, text: str) -> list[str]:
        import yake
        key_words = yake.KeywordExtractor(top=10, stopwords=None).extract_keywords(text)
        return [word[0] for word in key_words]

    @classmethod
    def __keybert(cls, text: str) -> list[str]:
        from keybert import KeyBERT
        kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        key_words = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english',
                                              highlight=False, top_n=10)
        return [word[0] for word in key_words]

    @classmethod
    def __summa(cls, text: str) -> list[str]:
        from summa import keywords
        keywords = keywords.keywords(text, scores=False)
        return keywords

    def __tfidf(self, text: str) -> list[str]:
        df = AdjustedVectorization([text]).tfidf()
        keywords = df.sort_values(axis=0)[-self.num_keywords:].index
        return list(keywords)

    def __tfidf_cluster(self) -> dict[str: set[str]]:  # TODO: set -> tuple
        """Group text togehter and use TFIDF scores to find the most important words.  Return a dictionary of keywords with
         the key being the cluster prediction and the value being the keywords"""
        separated_data = GroupLabeledTextData((self.labels, self.data)).separated_data
        keywords_dict = {}
        for label, data in separated_data:
            if self.label_occurrences[label] >= self.min_clust_size:
                df = AdjustedVectorization(data).tfidf()
                centroid = df.mean(axis=0)
                keywords = centroid.sort_values(axis=0)[-self.num_keywords:].index
                keywords_dict[label] = tuple(keywords)
        return keywords_dict

    def __LDA_cluster(self) -> dict[str: set[str]]: 
        """Use Latent Dirichlet Allocation to statistically find the most likely keywords.  Return a dictionary of keywords with
         the key being the cluster prediction and the value being the keywords"""
        from LDAKeywordExtraction import getKeywordsLDA
        separated_data = GroupLabeledTextData((self.labels, self.data)).separated_data
        keyDict = {}
        for x in range(len(separated_data)):
            label = separated_data[x][0]
            texts = separated_data[x][1][0:1000]
            keywords = getKeywordsLDA(texts, self.num_keywords)
            keyDict[label] = tuple(keywords)
        return keyDict   

    def __get_keywords(self, model) -> dict[str: set[str]]:
        """Select the keyword model based on the input"""
        if model == 'yake':
            return self.__most_frequent_keywords(self.__yake)
        elif model == 'keybert':
            return self.__most_frequent_keywords(self.__keybert)
        elif model == 'summa':
            return self.__most_frequent_keywords(self.__summa)
        elif model == 'tfidf':
            return self.__most_frequent_keywords(self.__tfidf)
        elif model == 'tfidf-cluster':
            return self.__tfidf_cluster()
        elif model == 'LDA':
            return self.__LDA_cluster()
        else:
            raise KeyError('Invalid keyword_model. Must be in {yake, keybert, summa, tfidf, tfidf-cluster}.')

    def __cluster_keywords(self, keyword_model, num_chars=1000) -> dict[str: list[str]]:
        """Get keywords for each document and cluster keywords of the same group together. Return a dictionary of keywords with
         the key being the cluster prediction and the value being the keywords"""
        keywords = {}
        for label, text in zip(self.labels, self.data):
            if self.label_occurrences[label] >= self.min_clust_size:
                try:
                    keywords[label] = keywords[label] + keyword_model(text[:num_chars])
                except KeyError:
                    keywords[label] = keyword_model(text[:num_chars])
        return keywords

    def __most_frequent_keywords(self, keyword_model) -> dict[str: set[str]]:
        """From all the keywords of the same group find the most frequent keywords. Return a dictionary of keywords with
         the key being the cluster prediction and the value being the keywords"""
        keywords_dict = self.__cluster_keywords(keyword_model)
        for (key, item) in keywords_dict.items():
            count_occurrences = Counter(item)
            keywords = sorted(count_occurrences, key=count_occurrences.get, reverse=True)
            keywords_dict[key] = set(keywords[:self.num_keywords])
        return keywords_dict