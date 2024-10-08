import cv2
import string
import nltk
import requests
import numpy as np
from nltk.corpus import stopwords
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from scipy.stats.contingency import association
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


def feature_to_target_corr(df, features, label, save_path):
    corr_values = {}
    p_values = {}
    for col in features:
        if df[col].nunique() == 2:
            corr = association(pd.crosstab(df[col], df[label]), method="cramer")
            pval = 0
        else:
            corr, pval = pointbiserialr(df[col].astype(float), df[label].astype(float))
        corr_values[col] = corr
        p_values[col] = pval

    sorted_values = sorted(zip(corr_values.values(), p_values.values(), corr_values.keys()), key=lambda x: abs(x[0]), reverse=True)
    corr_series = pd.Series([x[0] for x in sorted_values], index=[x[2] for x in sorted_values])

    # Create color mapping based on p-values
    vmax = max(p_values.values())
    cmap = sns.cm.rocket
    p_colors = [cmap(p_values[feature]/vmax) for feature in corr_series.index]

    # Create figure and axis objects
    fig, ax = plt.subplots()

    # Create bar plot of correlation values
    sns.barplot(x=corr_series.values, 
                y=corr_series.index,
                palette=p_colors, ax=ax)

    # Create a new axis for the colorbar
    norm = plt.Normalize(vmin=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Needed for ScalarMappable

    # Add the colorbar
    fig.colorbar(sm, ax=ax, label="p-value")

    # Set plot title and labels
    ax.set_title("Biserial Correlation between all features and Target Label", fontweight="bold")
    ax.set_xlabel("Correlation Coefficient", fontweight="bold")
    ax.set_ylabel("Feature Name", fontweight="bold")
    plt.tight_layout()

    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show the plot
    plt.show()


def read_image_from_url(url):
    response = requests.get(url)
    image_array = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image