import google.generativeai as genai
import pandas as pd
import json
import requests
from PIL import Image
from io import BytesIO
import time
from utils import evaluate_hals_preds
from constants import LABEL
import re
from prompts import *
import traceback
import numpy as np

API_KEY = "PUBLIC KEY"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')
counter = 1


def handle_prompt(prompt_func, row, img):
    sentence = row['sentence']
    pattern = r'<0x0A>|</s>'
    sentence = re.sub(pattern, ' ', sentence)
    if prompt_func.__name__ == 'LIST_SCORES_PROMPT':
        prompt = prompt_func(sentence, row['sentence_probes'])
    elif prompt_func.__name__ == 'LIST_SCORES_CLEAN_PROMPT':
        prompt = prompt_func(sentence)
    elif prompt_func.__name__ == 'EXPLICIT_FEATURE_ANNOTATION_PROMPT':
        prompt = prompt_func(sentence, row['sentence_probes'], row['sentence_POS_list'])
    elif prompt_func.__name__ == 'EXPLICIT_FEATURE_WA_POS_ANNOTATION_PROMPT':
        prompt = prompt_func(sentence, row['sentence_probes'])

    elif prompt_func.__name__ == 'RISK_WORD_IDENTIFICATION_PROMPT':
        prompt = prompt_func(sentence, row['sentence_probes'], row['sentence_POS_list'])
    elif prompt_func.__name__ == 'RISK_WORD_IDENTIFICATION_WA_POS_PROMPT':
        prompt = prompt_func(sentence, row['sentence_probes'])
    elif prompt_func.__name__ == 'RISK_WORD_IDENTIFICATION_CLEAN_PROMPT':
        prompt = prompt_func(sentence)

    elif prompt_func.__name__ == 'RISK_CLASSIFICATION_PROMPT':
        prompt = prompt_func(row['sentence_with_risk_words'])
    else:
        raise ValueError(f"got unexpected prompt func: {prompt_func.__name__}")
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()
    if 'RISK_WORD_IDENTIFICATION' in prompt_func.__name__:
        row['sentence_with_risk_words'] = response.text
    return response.text, row


def find_hallucination(row, prompt_funcs=[]):
    global counter
    try:
        img_path = row['image_link']
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content))
        for prompt_func in prompt_funcs:
            output, row = handle_prompt(prompt_func, row, img)
            if counter%5 == 0:   
                time.sleep(25)         
            # continuous_samples = np.random.normal(0, 10, 1)[0]
            # random_noise = int(np.round(continuous_samples))
            # time.sleep(max(0, 60 + random_noise))

            # else:
            #     continuous_samples = np.random.normal(0, 1, 1)[0]
            #     random_noise = int(np.round(continuous_samples))
            #     time.sleep(max(0, 5 + random_noise))
            counter += 1
            if counter%50 == 0:
                print(output)
            # print(output)
        return output
    except Exception as e:
        time.sleep(60)
        print(e)
        return "{'classification': -1}"


def parse_result(text):
    start_json = text.find('{')
    end_json = text.rfind('}') + 1
    
    if start_json == -1 or end_json == -1:
        return {'classification':-1}
    
    json_text = text[start_json:end_json]
    
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return {'classification':-1}


df = pd.read_pickle("/home/student/HallucinationsLLM/data/sentences_df_v2.pkl")

# df['llm_result'] = df.apply(find_hallucination, prompt_funcs=[LIST_SCORES_PROMPT], axis=1)
# df['llm_result'] = df.apply(find_hallucination, prompt_funcs=[EXPLICIT_FEATURE_ANNOTATION_PROMPT], axis=1)
# df['llm_result'] = df.apply(find_hallucination, prompt_funcs=[RISK_CLASSIFICATION_PROMPT], axis=1)

prompts_lists = [
                # [LIST_SCORES_CLEAN_PROMPT], 
                # [RISK_WORD_IDENTIFICATION_CLEAN_PROMPT, RISK_CLASSIFICATION_PROMPT], 
                # [EXPLICIT_FEATURE_ANNOTATION_PROMPT], 
                # [EXPLICIT_FEATURE_WA_POS_ANNOTATION_PROMPT],
                [RISK_WORD_IDENTIFICATION_WA_POS_PROMPT, RISK_CLASSIFICATION_PROMPT], 
                [RISK_WORD_IDENTIFICATION_PROMPT, RISK_CLASSIFICATION_PROMPT],  
                ]

for prompts_list in prompts_lists:
    print("----"* 50)
    print([prompt.__name__ for prompt in prompts_list])
    df['llm_result'] = df.apply(find_hallucination, prompt_funcs=prompts_list, axis=1)
    df['llm_result'] = df['llm_result'].apply(parse_result)
    df['llm_descision'] = df['llm_result'].apply(lambda llm_descision: llm_descision['classification'])
    filtered_df = df[df['llm_descision']!=-1]
    metric_eval = evaluate_hals_preds(filtered_df['llm_descision'].astype(int), filtered_df[LABEL].astype(int))
print("Finish Running!!!")

# filtered_df.to_pickle("llm_descision.pkl")


