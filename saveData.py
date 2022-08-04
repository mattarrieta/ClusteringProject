import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re

def save_folders(given_dir, main_folder, sub_folder, labels, data):
    """Saves file given the label, data and directory"""
    main_dir = rf'{os.getcwd()}\ClusteredData' if given_dir == 0 else given_dir
    for name, label, text in zip(sub_folder, labels, data):
        #print("Name:", name, "Label:", label, "Text:", text[0:10])
        #if(label not in labelDict):
        #    labelDict[label] = 1
        SaveData(rf'{main_dir}\{name}').make_new_dir()
        SaveData(rf'{main_dir}\{name}\{label}').write_new_file(text)

class SaveData:
    """ General-use class for saving single-computation data. Used to reduce run-time.  """
    def __init__(self, path):
        self.path = path

    def make_new_dir(self) -> None:
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def read_in_file(self) -> str:
        with open(self.path, encoding="mbcs") as file:
            return file.read()

    def write_new_file(self, text) -> str:
        with open(self.path, 'w', encoding="mbcs") as file:
            file.write(text)
            return text
