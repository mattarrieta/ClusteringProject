import os
import re
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
import time
from collections import Counter
import sys
sys.path.insert(0, r'C:\Users\Matthew Arrieta\Desktop\Project3Testing\TestFiles')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 


def preprocess(sentence, RemoveNums = False, stem = False, lem = False, stopwords_dict = None):
    """Preprocess text with options of removing numbers, stemming, and lemmatizing"""
    sentence=str(sentence)
    sentence = sentence.lower()
    #Remove HTML Tags
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    #Remove foregin characters
    only_ascii_encode = rem_url.encode('ascii',errors='ignore')
    only_ascii_decode = only_ascii_encode.decode()
    #Remove numbers
    if(RemoveNums):
        rem_num = re.sub('[0-9]+', '', only_ascii_decode)
    else:
        rem_num = only_ascii_decode
    #Remove Stopwords
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords_dict] #Can remove len(w) > 2 if want single and double letter tokens
    #Stem words
    if(stem):
        stem_words=[stemmer.stem(w) for w in filtered_words]
    else:
        stem_words = filtered_words
    #Lemmatize words
    if(lem):
        lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    else:
        lemma_words = stem_words
    return " ".join(lemma_words)

#Drop duplicate columns
def drop_duplicates(labeled_df):
    """Drop duplicate texts from data frames"""
    data_copy_df = labeled_df.copy()
    data_no_duplicates = data_copy_df.drop_duplicates(subset = ['text'])
    return data_no_duplicates

#Crop text to needed number of words
def crop_documents(labeled_df, document_cutoff_length):
    """Returns a df with documents the size of 'document_cutoff_length'"""
    labeled_df_copy = labeled_df.copy()
    labeled_df_copy['tokens_cropped'] = labeled_df_copy['text'].apply(lambda x: x.split()[:document_cutoff_length])
    labeled_df_copy['cropped_documents'] = labeled_df_copy['tokens_cropped'].apply(lambda x: ' '.join(x))
    labeled_df_copy = labeled_df_copy.drop(columns = ['tokens_cropped'])
    return labeled_df_copy

#Remove page numbers from file
def remove_page_nos(text):
    """Remove page numbers from texts"""
    new_text = re.subn(r'Page \d*', '', text)
    return new_text[0]
    
#Remove page numbers from text
def remove_metadata(text):
    """Remove metadata from text"""
    parts = text.split('Page 1\n')
    if len(parts)>1:
        return parts[1]
    else:
        return text


def get_corpus(labeled_df, on_entire_doc, document_cut_off):
    """Given a data frame get the text files of the dataframe and crop if needed"""
    if on_entire_doc == False:
        crop_documents_df = crop_documents(labeled_df, document_cut_off)
        corpus_to_train_on = crop_documents_df['cropped_documents'].to_list()
    else:
        corpus_to_train_on = labeled_df['text'].to_list()
    return corpus_to_train_on

def getText(path, knowLabels = True, preprocessData = False, separateOutliers = True, RemoveNums = False, stem = False, lem = False, extendStopWords = False):
    """Takes in data of a group of text files separated by labels and outputs the text and labels in separate lists.
       The input is a file path to a folder that contains subfolders labeled by the topic and each subfolder contains
        text files of that topic. The data can be processed if preprocessData = True, the outliers can be separate categories
        if separateOutliers are true and the rest of the parameters indicate preprocessing that can be done. """
    filenames = []
    file_path_l = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            filenames.append(filename)
            file_path_l.append(os.path.join(root, filename))


    text_l = []
    for file_path in file_path_l:
        filey = open(file_path,encoding="mbcs")
        text_l.append(filey.read())
        filey.close()


    if(extendStopWords):
        stop_words = stopwords.words('english')
        stop_words.extend(['url', 'xmp', 'url', 'pdfaproperty', 'etal','retrieved', 'pdfx', 'acrobat', 'pdf', 'biomed', 'arbortext', 'txt', 'crossmark', 'springerlink','crossmarkdomainexclusive', 'distiller', 'itext', 'producer', 'com', 'doi', 'prism', 'author', 'prism'])
        stopwords_dict = Counter(stop_words)
    else:
        stop_words = stopwords.words('english')
        stopwords_dict = Counter(stop_words)

    outlier_num = 0
    types_l = []
    if(knowLabels):
        if(separateOutliers):
            for namey in file_path_l:
                splitted = namey.split('\\')
                if splitted[len(splitted) - 2] == 'Outliers':
                    types_l.append(f"Outlier-{outlier_num}")
                    outlier_num += 1
                else:
                    types_l.append(splitted[len(splitted) - 2])
        else:
            for namey in file_path_l:
                splitted = namey.split('\\')
                types_l.append(splitted[len(splitted) - 2])
    else:
        types_l = None
    cleanText = []
    if(preprocessData):
        for i in range(len(text_l)):
            cleanText.append(preprocess(remove_metadata(remove_page_nos(text_l[i])), RemoveNums = RemoveNums, stem = stem, lem = lem, stopwords_dict = stopwords_dict))
    else:
        for i in range(len(text_l)):
            cleanText.append((remove_metadata(remove_page_nos(text_l[i]))))
    return cleanText, types_l, filenames

def getDataFrame(path, cut = 0, preprocess = False, separateOutliers = True, RemoveNums = False, stem = False, lem = False, extendStopWords = False):
    """Takes in data of a group of text files separated by labels and outputs a dataframe containg labels and text.
       The input is a file path to a folder that contains subfolders labeled by the topic and each subfolder contains
        text files of that topic. The data can be processed if preprocessData = True, the outliers can be separate categories
        if separateOutliers are true and the rest of the parameters indicate preprocessing that can be done. """
    cleanText, types_l = getText(path, preprocess = preprocess, separateOutliers= separateOutliers, RemoveNums= RemoveNums, stem = stem, lem = lem, extendStopWords= extendStopWords)

    d = {"labels": types_l, "text": cleanText}
    textInput = pd.DataFrame(data = d)

    if(cut != 0):
        textInputCut = get_corpus(textInput, False, cut)
        d2 = {"labels": types_l, "text": textInputCut}
        textInput = pd.DataFrame(data = d2)

    return textInput