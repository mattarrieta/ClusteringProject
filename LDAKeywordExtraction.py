from gensim.models import LdaModel, HdpModel
from gensim import corpora
import pandas as pd


def order_subset_by_coherence(dirichlet_model, bow_corpus, num_topics=10, num_keywords=10):
    """
    Orders topics based on their average coherence across the corpus

    Parameters
    ----------
        dirichlet_model : gensim.models.type_of_model
        bow_corpus : list of lists (contains (id, freq) tuples)
        num_topics : int (default=10)
        num_keywords : int (default=10)

    Returns
    -------
        ordered_topics, ordered_topic_averages: list of lists and list
    """
    if type(dirichlet_model) == LdaModel:
        shown_topics = dirichlet_model.show_topics(num_topics=num_topics, 
                                                   num_words=num_keywords,
                                                   formatted=False)
    elif type(dirichlet_model)  == HdpModel:
        shown_topics = dirichlet_model.show_topics(num_topics=150, # return all topics
                                                   num_words=num_keywords,
                                                   formatted=False)
    model_topics = [[word[0] for word in topic[1]] for topic in shown_topics]
    topic_corpus = dirichlet_model.__getitem__(bow=bow_corpus, eps=0) # cutoff probability to 0 

    topics_per_response = [response for response in topic_corpus]
    flat_topic_coherences = [item for sublist in topics_per_response for item in sublist]

    significant_topics = list(set([t_c[0] for t_c in flat_topic_coherences])) # those that appear
    topic_averages = [sum([t_c[1] for t_c in flat_topic_coherences if t_c[0] == topic_num]) / len(bow_corpus) \
                      for topic_num in significant_topics]

    topic_indexes_by_avg_coherence = [tup[0] for tup in sorted(enumerate(topic_averages), key=lambda i:i[1])[::-1]]

    significant_topics_by_avg_coherence = [significant_topics[i] for i in topic_indexes_by_avg_coherence]
    ordered_topics = [model_topics[i] for i in significant_topics_by_avg_coherence][:num_topics] # limit for HDP

    ordered_topic_averages = [topic_averages[i] for i in topic_indexes_by_avg_coherence][:num_topics] # limit for HDP
    ordered_topic_averages = [a/sum(ordered_topic_averages) for a in ordered_topic_averages] # normalize HDP values

    return ordered_topics, ordered_topic_averages

def getKeywordsLDA(group, numKeywords):

    for x in range(len(group)):
        group[x] = group[x].split()
    dirichlet_dict = corpora.Dictionary(group)
    bow_corpus = [dirichlet_dict.doc2bow(text) for text in group]

    num_topics = 1
    num_keywords = numKeywords

    dirichlet_model = LdaModel(corpus=bow_corpus, id2word=dirichlet_dict, num_topics=num_topics, update_every=1, chunksize=len(bow_corpus), passes=20, alpha='auto')

    ordered_topics, ordered_topic_averages = \
        order_subset_by_coherence(dirichlet_model=dirichlet_model,
                                bow_corpus=bow_corpus, 
                                num_topics=num_topics,
                                num_keywords=num_keywords)

    keywords = []
    for i in range(num_topics):
        # Find the number of indexes to select, which can later be extended if the word has already been selected
        selection_indexes = list(range(int(round(num_keywords * ordered_topic_averages[i]))))
        if selection_indexes == [] and len(keywords) < num_keywords: 
            # Fix potential rounding error by giving this topic one selection
            selection_indexes = [0]
                
        for s_i in selection_indexes:
            if ordered_topics[i][s_i] not in keywords:
                keywords.append(ordered_topics[i][s_i])
            else:
                selection_indexes.append(selection_indexes[-1] + 1)

    # Fix for if too many were selected
    keywords = keywords[:num_keywords]
    return keywords

def keywordsLDA(separated_data, numKeywords, shortText= True):
    """Given a list of list of documents grouped by cluster return a dictionary of keywords with the key being the 
        cluster prediction and the value being the keywords"""
    keyDict = {}
    for x in range(len(separated_data)):
        label = separated_data[x][0]
        if(shortText):
            texts = separated_data[x][1][0:1000]
        else:
            texts = separated_data[x][1]
        keywords = getKeywordsLDA(texts, numKeywords)
        keyDict[label] = tuple(keywords)
    return keyDict
