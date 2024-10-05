import string
import nltk
from nltk.corpus import stopwords
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def prepare_stopwords_set():
    stop_words = set(stopwords.words('english'))
    for punc in string.punctuation:
        stop_words.add(punc)
    return stop_words


def evaluate_hals_preds(preds, labels, plot=True):
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    if plot:
        print("Preds Frequency")
        print(pd.Series(preds).value_counts())

        print("Actual Frequency")
        print(pd.Series(labels).value_counts())

        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        print(results)
    return results