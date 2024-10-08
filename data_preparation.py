import re
import torch
import cv2
import ast
import statistics
import pandas as pd
from constants import *
from utils import read_image_from_url
from scipy.stats import entropy
from ultralytics import YOLO
from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
yolo_model = YOLO("yolov10x.pt")


def extract_object_num(image_link):
    def predict(chosen_model, img, classes=[], conf=0.5):
        if classes:
            results = chosen_model.predict(img, classes=classes, conf=conf)
        else:
            results = chosen_model.predict(img, conf=conf)

        classes_num = results[0].boxes.cls
        classes_names = results[0].names
        return [classes_names[int(num)] for num in classes_num]

    image = read_image_from_url(image_link)
    results = predict(yolo_model, image, classes=[], conf=0.5)
    return len(results)


def extract_rgb_means(image_link):
    image = read_image_from_url(image_link)
    (B, G, R) = cv2.split(image)

    mean_B = np.mean(B)
    mean_G = np.mean(G)
    mean_R = np.mean(R)
    return pd.Series([mean_B, mean_G, mean_R, (mean_B + mean_G + mean_R) / 3])


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


def get_pos_tags(words_logits_mapping):
    tokens = [maping[0] for maping in words_logits_mapping]
    tagged = nltk.pos_tag(tokens)
    return tagged


def get_pos_by_sentence(description, pos_mapping):
    pattern = r'<0x0A>|</s>'
    cleaned_description = re.sub(pattern, ' ', description)
    sentences = cleaned_description.split(".")[:-1]
    sentences_pos = []
    counter = 0 
    for sentence in sentences:
        words = sentence.split()
        sentence_pos = set()
        for j, word in enumerate(words):
            sentence_pos.add(pos_mapping[j + counter][POS])
            assert word.strip() == pos_mapping[j + counter][WORD], f"Unmatching words in POS: {word.strip()}, {pos_mapping[j + counter][WORD]}"
        sentences_pos.append(list(sentence_pos))
        counter += len(words)   
    return sentences_pos


def get_probes_by_sentence(description, words_logits_mapping, logits):
    pattern = r'<0x0A>|</s>'
    cleaned_description = re.sub(pattern, ' ', description)
    sentences = cleaned_description.split(".")[:-1]
    sentences_probs = []
    counter = 0 
    for sentence in sentences:
        words = sentence.split()
        sentence_probes = []
        for j, word in enumerate(words):
            assert word.strip() == words_logits_mapping[j + counter][WORD], f"Unmatching words in POS: {word.strip()}, {words_logits_mapping[j + counter][WORD]}"
            word_logits_map = words_logits_mapping[j]
            prob_idx = word_logits_map[INDICES][0]
            sentence_probes.append(logits[prob_idx][INDICES][logits[prob_idx][WORD].strip()])
        sentences_probs.append(list(sentence_probes))
        counter += len(words)   
    return sentences_probs


def get_entropy_by_sentence(description, words_logits_mapping, logits):
    pattern = r'<0x0A>|</s>'
    cleaned_description = re.sub(pattern, ' ', description)
    sentences = cleaned_description.split(".")[:-1]
    sentences_entropies = []
    counter = 0 
    for sentence in sentences:
        words = sentence.split()
        sentence_entropies = []
        for j, word in enumerate(words):
            assert word.strip() == words_logits_mapping[j + counter][WORD], f"Unmatching words in POS: {word.strip()}, {words_logits_mapping[j + counter][WORD]}"
            word_logits_map = words_logits_mapping[j]
            prob_idx = word_logits_map[INDICES][0]
            sentence_entropies.append(entropy(list(logits[prob_idx][1].values()), base=2))
        sentences_entropies.append(list(sentence_entropies))
        counter += len(words)   
    return sentences_entropies



def sentence_length(sentence):
    words = sentence.split()
    return len(words)


def mean_pos_prob(sentence_pos):
    probs = []
    for pos in sentence_pos:
        probs.append(POS_HAL_PROBS[pos])
    return sum(probs) / len(probs)


def max_pos_prob(sentence_pos):
    probs = []
    for pos in sentence_pos:
        probs.append(POS_HAL_PROBS[pos])
    return max(probs)


def min_pos_prob(sentence_pos):
    probs = []
    for pos in sentence_pos:
        probs.append(POS_HAL_PROBS[pos])
    return min(probs)


def median_pos_prob(sentence_pos):
    probs = []
    for pos in sentence_pos:
        probs.append(POS_HAL_PROBS[pos])
    return statistics.median(probs)


def image_embedding(image_link):
    image = Image.open(requests.get(image_link, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")
    image_features = clip_model.get_image_features(**inputs)
    return image_features


def sentence_embedding(sentence):
    pattern = r'<0x0A>|</s>'
    cleaned_sentence = re.sub(pattern, ' ', sentence)
    cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()
    inputs = tokenizer([cleaned_sentence], padding=True, return_tensors="pt")
    text_features = clip_model.get_text_features(**inputs)
    return text_features


def sentence_image_similarity(image_vec, sentence_vec):
    return float(torch.nn.functional.cosine_similarity(image_vec, sentence_vec))


def prepare_data(data_path):
    df = pd.read_excel(data_path, index_col=0)
    df['logits'] = df['logits'].apply(lambda x: ast.literal_eval(x))
    df[['mean_b', 'mean_g', 'mean_r', 'mean_rbg']] = df.apply(lambda row: extract_rgb_means(row['image_link']), axis=1)
    df['objects_num'] = df['image_link'].apply(extract_object_num)
    df['words_logits_mapping'] = df.apply(lambda row: map_words_logits(row['logits'], row['description']), axis=1)
    df['POS'] = df['words_logits_mapping'].apply(lambda words_logits_mapping: get_pos_tags(words_logits_mapping))
    df['sentence_POS'] = df.apply(lambda row: get_pos_by_sentence(row['description'], row['POS']), axis=1)
    df['sentence_probes'] = df.apply(lambda row: get_probes_by_sentence(row['description'], row['words_logits_mapping'], row['logits']), axis=1)
    df['sentence_entropy'] = df.apply(lambda row: get_entropy_by_sentence(row['description'], row['words_logits_mapping'], row['logits']), axis=1)
    df['sentence_normalized_index'] = df['hallucinations'].apply(sentence_normalized_index)
    df['sentence_contains_hedges'] = df['hedges'].apply(sentence_contains_hedges)
    df['sentence'] = df['description'].apply(extract_sentence)
    df['sentences_labels'] = df['hallucinations'].apply(extract_sentence_hals_labels)
    df['image_embedding'] = df['image_link'].apply(image_embedding)
    return df


def validate_list_lengths(df, columns):
    return df[columns].applymap(len).nunique(axis=1).eq(1).all()


def prepare_sentences_df(data_path):
    df = prepare_data(data_path)
    features = [feature for feature in SENTENCE_COLUMNS if feature in df.columns]
    assert validate_list_lengths(df, features) == True
    sentences_df = df[[col for col in RELEVANT_COLUMNS if col in df.columns]].explode(column=features).reset_index(drop=True)
    sentences_df['sentence_len'] = sentences_df['sentence'].apply(sentence_length)

    # Logits (probes)
    sentences_df['median_sentence_probes'] = sentences_df['sentence_probes'].apply(lambda x: statistics.median(x))
    sentences_df['min_sentence_probes'] = sentences_df['sentence_probes'].apply(lambda x: min(x))
    sentences_df['max_sentence_probes'] = sentences_df['sentence_probes'].apply(lambda x: max(x))
    sentences_df['mean_sentence_probes'] = sentences_df['sentence_probes'].apply(lambda x: statistics.mean(x))

    # Logits (entropy)
    sentences_df['median_sentence_entropy'] = sentences_df['sentence_entropy'].apply(lambda x: statistics.median(x))
    sentences_df['min_sentence_entropy'] = sentences_df['sentence_entropy'].apply(lambda x: min(x))
    sentences_df['max_sentence_entropy'] = sentences_df['sentence_entropy'].apply(lambda x: max(x))
    sentences_df['mean_sentence_entropy'] = sentences_df['sentence_entropy'].apply(lambda x: statistics.mean(x))

    # POS probs for hallucinations
    sentences_df['min_pos_prob'] = sentences_df['sentence_POS'].apply(min_pos_prob)
    sentences_df['mean_pos_prob'] = sentences_df['sentence_POS'].apply(mean_pos_prob)
    sentences_df['max_pos_prob'] = sentences_df['sentence_POS'].apply(max_pos_prob)
    sentences_df['median_pos_prob'] = sentences_df['sentence_POS'].apply(median_pos_prob)

    # CLIP 
    sentences_df['sentence_embedding'] = sentences_df['sentence'].apply(sentence_embedding)
    sentences_df['sentence_image_similarity'] = sentences_df.apply(lambda row: sentence_image_similarity(row['image_embedding'], row['sentence_embedding']), axis=1)

    sentences_df.to_pickle('sentences_df_clip-vit-large-patch14.pkl')
    return sentences_df



