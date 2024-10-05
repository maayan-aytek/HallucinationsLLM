import re
import ast
import nltk
import string
import random
import statistics
import pandas as pd
from evaluation import evaluate_hals_preds
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


WORD = 0
INDICES = 1


def prepare_stopwords_set():
    stop_words = set(stopwords.words('english'))
    for punc in string.punctuation:
        stop_words.add(punc)
    return stop_words


def map_words_logits(logits, description):
    pattern = r'<0x0A>|</s>'
    cleaned_description = re.sub(pattern, ' ', description)
    cleaned_description = re.sub(r'\s+', ' ', cleaned_description).strip()
    words = cleaned_description.split(" ")
    words_logits_indices = []
    idx = 0 
    words_token = ""
    idx_list = []
    flag = False
    for i, (token, prob_dict) in enumerate(logits):
        if ">" == token[-1]:
            flag = False
            continue
        if "<" in token:
            flag = True
        if flag:
            continue
        
        if token.strip() == ".":
            continue
        if words[idx][-1] == "." and token.strip()[-1] != ".":
            word = words[idx][:-1]
        else:
            word = words[idx]
        if token.strip() == word:
            words_logits_indices.append((word.replace(".", ""), [i]))
            idx += 1 
        else:
            words_token += token.strip()
            idx_list.append(i)
            if words_token == word:
                words_logits_indices.append((word.replace(".", ""), idx_list))
                words_token = ""
                idx_list = []
                idx += 1 
    return words_logits_indices


def extract_sentence_hals_labels(hallucinations):
    sentences = hallucinations.split(".")[:-1]
    labels = []
    for sentence in sentences:
        if "[" in sentence:
            labels.append(1)
        else:
            labels.append(0)
    return labels


def sentence_normalized_index(hallucinations):
    sentences = hallucinations.split(".")[:-1]
    num_sentences = len(sentences)
    indices = []
    for i in range(num_sentences):
        indices.append(i / num_sentences)
    return indices


def sentence_contains_hedges(hedges):
    sentences = hedges.split(".")[:-1]
    contains_hedges = []
    for sentence in sentences:
        if "[" in sentence:
            contains_hedges.append(1)
        else:
            contains_hedges.append(0)
    return contains_hedges


def extract_sentence(descriptions):
    sentences = descriptions.split(".")[:-1]
    return sentences


def hallucinations_tagging(description, logits, words_logits_mapping, stop_words_set, agg='median', threshold_dict = {1:0.23, 2:0.37, 3:0.24, 4:0.26}, sample_seq=False):
    pattern = r'<0x0A>|</s>'
    cleaned_description = re.sub(pattern, ' ', description)
    sentences = cleaned_description.split(".")[:-1]
    seq_lens = [1, 2, 3, 4]
    probs = [0.76, 0.145, 0.05, 0.045]
    if sample_seq:
        seq_lens = random.choices(seq_lens, weights=probs, k=1)
    counter = 0 
    sentences_hals_indices = []
    for sentence in sentences:
        words = sentence.split()
        cleaned_words_indices = [(j + counter, word) for j, word in enumerate(words) if word not in stop_words_set]
        contains_hal = []
        for seq_len in seq_lens:
            if seq_len <= len(cleaned_words_indices):
                for i in range(len(cleaned_words_indices) - seq_len + 1):
                    words_seq = cleaned_words_indices[i:i + seq_len]
                    seq_probs = []
                    for index, word in words_seq:
                        words_logits_map = words_logits_mapping[index]
                        assert word.strip() == words_logits_map[WORD], f"Unmatching words: {word.strip()}, {words_logits_map[WORD]}"
                        prob_idx = words_logits_map[INDICES][0]
                        seq_probs.append(logits[prob_idx][INDICES][logits[prob_idx][WORD].strip()])
                if agg == "median":
                    result = statistics.median(seq_probs)
                elif agg == "mean":
                    result = statistics.mean(seq_probs)
                else:
                    raise ValueError(f"Unkown aggeration: {agg}")
                if result < threshold_dict[seq_len]:
                    contains_hal.append((i, seq_len))
        sentences_hals_indices.append(contains_hal)
        counter += len(words)
    sentences_hals_preds = [1 if len(setnence_hals_indices) > 0 else 0 for setnence_hals_indices in sentences_hals_indices]
    return pd.Series([sentences_hals_preds, sentences_hals_indices])


def validate_list_lengths(df, columns):
    return df[columns].applymap(len).nunique(axis=1).eq(1).all()


if __name__ == "__main__":
    RELEVANT_COLUMNS = ['description', 'hallucinations', 'logits', 'words_logits_mapping', 'sentence', 'sentence_normalized_index', 'sentences_labels', 'sentences_preds', 'sentence_contains_hedges']
    SENTENCE_FEATURES = ['sentence_normalized_index', 'sentences_labels', 'sentences_preds', 'sentence', 'sentence_contains_hedges']

    stop_words_set = prepare_stopwords_set()
    custom_stop_words = (stop_words_set - {'above', 'all', 'any', 'are', "aren't", 'between', 'below', 'before', 'both', 'down', 'few',
                                        'each', 'from', 'further', 'he', 'her', 'here', 'hers', 'him', 'his', 'in', 'into',  'is', 'isn',
                                            "isn't", 'it', "it's", 'its', 'itself', 'most', 'no', 'not', 'on', 'of', 'off', 'only', 'once', 'other', 'over', 'out',
                                            'some', 'their', 'theirs', 'them', 'they', 'there', 'to', 'too', 'was', "wasn't", 'we', 'were', "weren't", ']', '['}).union(
                                        {'although', 'as', 'because', 'so', 'supposing', 'suggesting', 'additionally', 'that', 'than', 'though', 'till',
                                        'unless', 'until', 'when', 'whenever', 'accordingly', 'also', 'consequently', 'conversely', 'furthermore', 'finally', 'hence',
                                        'however', 'indeed', 'likewise', 'meanwhile', 'moreover', 'nevertheless', 'nonetheless', 'otherwise', 'similarly', 'therefore', 'thus'
                                        'whereas', 'wherever', 'whether', 'while', 'either', 'neither', 'but', 'yet'})

    df = pd.read_excel("/home/student/HallucinationsLLM/data/team5_clean_dataset.xlsx", index_col=0)

    df['logits'] = df['logits'].apply(lambda x: ast.literal_eval(x))
    df['words_logits_mapping'] = df.apply(lambda row: map_words_logits(row['logits'], row['description']), axis=1)
    df['sentence_normalized_index'] = df['hallucinations'].apply(sentence_normalized_index)
    df['sentence_contains_hedges'] = df['hedges'].apply(sentence_contains_hedges)
    df['sentence'] = df['description'].apply(extract_sentence)

    df['sentences_labels'] = df['hallucinations'].apply(extract_sentence_hals_labels)
    df[['sentences_preds', 'sentences_preds_indices']] = df.apply(lambda row: hallucinations_tagging(row['description'], row['logits'], row['words_logits_mapping'],
                                                                                                    custom_stop_words, agg='median', sample_seq=True), axis=1)

    
    sentences_df = df[RELEVANT_COLUMNS].explode(column=SENTENCE_FEATURES).reset_index(drop=True)

    evaluate_hals_preds(sentences_df['sentences_preds'].tolist(), sentences_df['sentences_labels'].tolist())