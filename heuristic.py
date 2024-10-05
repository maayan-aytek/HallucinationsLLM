import re
import pickle
import numpy as np
import random
import statistics
import pandas as pd
from utils import evaluate_hals_preds
from constants import *
from data_preparation import prepare_data, prepare_sentences_df


def hallucinations_tagging(description, logits, words_logits_mapping, POS_mapping, stop_words_set, agg='median', threshold_dict = {1:0.23, 2:0.37, 3:0.24, 4:0.26}, sample_seq=False):
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
        contains_cd = False
        words = sentence.split()
        cleaned_words_indices = [(j + counter, word) for j, word in enumerate(words) if word not in stop_words_set]
        contains_hal = []
        for seq_len in seq_lens:
            if seq_len <= len(cleaned_words_indices):
                for i in range(len(cleaned_words_indices) - seq_len + 1):
                    words_seq = cleaned_words_indices[i:i + seq_len]
                    seq_probs = []
                    for index, word in words_seq:
                        word_logits_map = words_logits_mapping[index]
                        word_pos = POS_mapping[index][POS]
                        assert word.strip() == word_logits_map[WORD], f"Unmatching words: {word.strip()}, {word_logits_map[WORD]}"
                        assert word.strip() == POS_mapping[index][WORD], f"Unmatching words in POS: {word.strip()}, {POS_mapping[index][WORD]}"
                        if word_pos == 'CD':
                            contains_cd = True
                        prob_idx = word_logits_map[INDICES][0]
                        seq_probs.append(logits[prob_idx][INDICES][logits[prob_idx][WORD].strip()])
                if agg == "median":
                    result = statistics.median(seq_probs)
                elif agg == "mean":
                    result = statistics.mean(seq_probs)
                else:
                    raise ValueError(f"Unkown aggeration: {agg}")
                if result < threshold_dict[seq_len] or contains_cd:
                    contains_hal.append((i, seq_len))
        sentences_hals_indices.append(contains_hal)
        counter += len(words)
    sentences_hals_preds = [1 if len(setnence_hals_indices) > 0 else 0 for setnence_hals_indices in sentences_hals_indices]
    return pd.Series([sentences_hals_preds, sentences_hals_indices])



def hallucinations_tagging_with_pos(description, logits, words_logits_mapping, POS_mapping, stop_words_set, probes_agg='median', pos_agg='max', words_threshold_dict = {1:0.23, 2:0.37, 3:0.24, 4:0.26}, pos_threshold=0.8, condition="and", sample_seq=False):
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
                    seq_words_probs = []
                    seq_pos_probes = []
                    for index, word in words_seq:
                        word_logits_map = words_logits_mapping[index]
                        word_pos = POS_mapping[index][POS]
                        assert word.strip() == word_logits_map[WORD], f"Unmatching words: {word.strip()}, {word_logits_map[WORD]}"
                        assert word.strip() == POS_mapping[index][WORD], f"Unmatching words in POS: {word.strip()}, {POS_mapping[index][WORD]}"
                        
                        prob_idx = word_logits_map[INDICES][0]
                        seq_words_probs.append(logits[prob_idx][INDICES][logits[prob_idx][WORD].strip()])
                        seq_pos_probes.append(POS_HAL_PROBS[word_pos])
                if probes_agg == "median":
                    words_probes_agg = statistics.median(seq_words_probs)
                elif probes_agg == "mean":
                    words_probes_agg = statistics.mean(seq_words_probs)
                else:
                    raise ValueError(f"Unkown aggeration: {probes_agg}")

                if pos_agg == "max":
                    pos_probes_agg = max(seq_pos_probes)
                elif pos_agg == "mean":
                    pos_probes_agg = statistics.mean(seq_pos_probes)
                elif pos_agg == "median":
                    pos_probes_agg = statistics.median(seq_pos_probes)
                else:
                    raise ValueError(f"Unkown aggeration: {probes_agg}")

                if condition == "or":
                    if words_probes_agg < words_threshold_dict[seq_len] or pos_probes_agg > pos_threshold:
                        contains_hal.append((i, seq_len))
                else:
                    if words_probes_agg < words_threshold_dict[seq_len] and pos_probes_agg > pos_threshold:
                        contains_hal.append((i, seq_len))
        sentences_hals_indices.append(contains_hal)
        
        counter += len(words)
    sentences_hals_preds = [1 if len(setnence_hals_indices) > 0 else 0 for setnence_hals_indices in sentences_hals_indices]
    return sentences_hals_preds


if __name__ == "__main__":
    path = "/home/student/HallucinationsLLM/data/team5_clean_dataset.xlsx"
    df = prepare_data(path)
    df[['sentences_preds', 'sentences_preds_indices']] = df.apply(lambda row: hallucinations_tagging(row['description'], row['logits'], row['words_logits_mapping'], row['POS'],
                                                                                                    STOP_WORDS, agg='median', sample_seq=True), axis=1)

    sentences_df = df[['sentences_preds', 'sentences_labels']].explode(['sentences_preds', 'sentences_labels'])


    # probes_aggs = ['median', 'mean']
    # pos_aggs = ['max', 'mean', 'median']
    # words_threshold_dicts = [{1:0.23, 2:0.37, 3:0.24, 4:0.26}, {1:0.13, 2:0.37, 3:0.22, 4:0.25}]
    # pos_thresholds = np.arange(0.1, 0.8, 0.05).tolist()
    # conditions = ["and", "or"]
    # sample_seqs = [True, False]
    # results_dict = {}
    # for probe_agg in probes_aggs:
    #     for pos_agg in pos_aggs:
    #         for words_threshold_dict in words_threshold_dicts:
    #             for pos_threshold in pos_thresholds:
    #                 for condition in conditions:
    #                     for sample_seq in sample_seqs:
    #                         sentences_hals_preds_series = df.apply(lambda row: hallucinations_tagging_with_pos(row['description'], row['logits'], row['words_logits_mapping'], row['POS'],
    #                                                                     STOP_WORDS, probes_agg=probe_agg, pos_agg=pos_agg, words_threshold_dict = words_threshold_dict,
    #                                                                     pos_threshold=pos_threshold, condition=condition, sample_seq=sample_seq), axis=1)
    #                         exp_str = f"probe_agg={probe_agg}_pos_agg={pos_agg}_words_threshold_dict={words_threshold_dict}_pos_threshold={pos_threshold}_condition={condition}_sample_seq={sample_seq}"
    #                         results_dict[exp_str] = evaluate_hals_preds(sentences_hals_preds_series.explode().tolist(), sentences_df['sentences_labels'].tolist(), plot=False)


    # results_dict = {k: v for k, v in sorted(results_dict.items(), key=lambda item: item[1]['f1_score'], reverse=True)}
    # print(results_dict)
    # with open('results_dict.pickle', 'wb') as f:
    #     pickle.dump(results_dict, f)
    evaluate_hals_preds(sentences_df['sentences_preds'].tolist(), sentences_df['sentences_labels'].tolist())

    # with open('results_dict.pickle', 'rb') as f:
    #     loaded_dict = pickle.load(f)
    # a = 2