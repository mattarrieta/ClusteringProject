import os
import pandas as pd


def doc2vec_train():
    """ Trains and saves doc2vec-train-model. """
    import gensim.models
    import gensim.downloader
    from gensim.models.doc2vec import Doc2Vec
    training_data = [d for d in gensim.downloader.load('text8')]

    def tagged_document(list_of_list_of_words):
        for idx, list_of_words in enumerate(list_of_list_of_words):
            yield gensim.models.doc2vec.TaggedDocument(list_of_words, [idx])

    data_for_training = list(tagged_document(training_data))
    model = Doc2Vec(epochs=30)
    model.build_vocab(data_for_training)
    model.train(data_for_training, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(rf'{os.getcwd()}\doc2vec-train-model')
    return model


def bert_train(model_type, data, labels=None) -> pd.DataFrame:
    """ Trains and saves bert model_type for given label and data. """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_type)
    bert_df = pd.DataFrame(model.encode(data, show_progress_bar=True), index=labels)
    bert_df.to_csv(rf'{os.getcwd()}\BertModels\{model_type}', index=False)
    return bert_df