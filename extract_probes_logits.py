import pandas as pd
import numpy as np
from PIL import Image
import requests
import torch
from transformers import (BitsAndBytesConfig, AutoProcessor,
                          LlavaForConditionalGeneration, GenerationConfig)



model_name = 'llava-hf/llava-1.5-7b-hf'
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.bfloat16)

model = LlavaForConditionalGeneration.from_pretrained(model_name,
                                                      quantization_config=quantization_config)
processor = AutoProcessor.from_pretrained(model_name)

df = pd.read_excel(f'/home/student/HallucinationsLLM/team5_project_data.xlsx', index_col=0)


def generate(image, prompt, processor, model, temperature=1.0):
    generation_config = GenerationConfig(max_new_tokens=200,
                                         do_sample=True,
                                         temperature=temperature,
                                         output_scores=True,
                                         return_dict_in_generate=True)
    device = model.device
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generate_outputs = model.generate(**inputs, generation_config=generation_config)
    generate_outputs['scores'] = [x*temperature for x in generate_outputs['scores']]
    return generate_outputs


def extract_logits(processor, generate_outputs, min_prob=1e-3):
    scores_len = len(generate_outputs['scores'])
    generated_ids = generate_outputs['sequences'].tolist()[0][-scores_len:]
    generated_scores = []
    for i, token_id in enumerate(generated_ids):
        token = processor.tokenizer._convert_id_to_token(token_id).replace("▁", " ")
        logits = torch.softmax(generate_outputs['scores'][i], dim=-1)[0].tolist()
        logits = {t: p for t, p in enumerate(logits) if (p > min_prob or t == token_id)}
        logits = {processor.tokenizer._convert_id_to_token(t).replace("▁", " "): p for t, p in logits.items()}
        generated_scores.append((token, logits))
    generated_text = ''.join([x for x, _ in generated_scores])

    logits = []
    for token, tokens_dict in generated_scores:
        logits.append(tokens_dict)
    return generated_text, logits


def extract_probes_logits(image_url, probe, temp):
    prompt = f"USER: <image>\nAccording to the image, is the following sentence correct? answer yes/no only: {probe}\nASSISTANT:"
    image = Image.open(requests.get(image_url, stream=True).raw)
    generate_outputs = generate(image, prompt, processor, model, temperature=temp)
    generated_text, generated_logits = extract_logits(processor, generate_outputs, min_prob=1e-3)
    generated_text_lower = generated_text.lower()
    if "yes" in generated_text_lower and "no" not in generated_text_lower:
        pred = "true"
    elif "no" in generated_text_lower:
        pred = "false"
    else:
        pred = ""
        print(f"unrecognised answer: {generated_text}, {image_url}")
    for logits_dict in generated_logits:
        lower_keys = [key.lower().strip() for key in logits_dict]
        if "yes" in lower_keys and "no" in lower_keys:
            sum = 0
            yes_prob = 0
            for key, val in logits_dict.items():
                if "yes" in key.lower():
                    sum += val
                    yes_prob = val
                if "no" in key.lower():
                    sum += val
            yes_prob /= sum
    return pd.Series([pred, max(yes_prob, 1-yes_prob)])


df[['pred_1', 'pred_1_prob']] = df.apply(lambda row: extract_probes_logits(row['image_link'], row['probe_1'], row['temperature']), axis=1)
print("finished 1")
df[['pred_2', 'pred_2_prob']] = df.apply(lambda row: extract_probes_logits(row['image_link'], row['probe_2'], row['temperature']), axis=1)
print("finished 2")
df[['pred_3', 'pred_3_prob']] = df.apply(lambda row: extract_probes_logits(row['image_link'], row['probe_3'], row['temperature']), axis=1)
print("finished 3")
df[['pred_4', 'pred_4_prob']] = df.apply(lambda row: extract_probes_logits(row['image_link'], row['probe_4'], row['temperature']), axis=1)

df.to_excel('result.xlsx')