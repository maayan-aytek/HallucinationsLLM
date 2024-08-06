from PIL import Image
import requests
import torch
from transformers import (BitsAndBytesConfig, AutoProcessor,
                          LlavaForConditionalGeneration, GenerationConfig)

model_name = 'llava-hf/llava-1.5-7b-hf'
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = LlavaForConditionalGeneration.from_pretrained(model_name, quantization_config=quantization_config)
processor = AutoProcessor.from_pretrained(model_name)


def extract_logits(processor, image_url, generated_description, min_prob=1e-3):
    image = Image.open(requests.get(image_url, stream=True).raw)
    instruction = "Please provide a thorough description of this image"
    prompt = f"USER: <image>\n{instruction}\nASSISTANT:"
    
    tokenized_prompt_input = processor.tokenizer.tokenize(prompt + generated_description)
    tokenized_prompt = processor.tokenizer.tokenize(prompt)
    description_length = len(tokenized_prompt_input) - len(tokenized_prompt) + 1
    tokenized_description = tokenized_prompt_input[-description_length+1:]

    inputs = processor(text=prompt+generated_description, images=image, return_tensors="pt")

    with torch.no_grad():  # Ensure no gradients are computed to save memory
        forward_outputs = model(**inputs, return_dict=True)
        forward_logits = forward_outputs[1][0][-description_length:]
    
    scores_len = len(forward_logits)
    generated_ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in tokenized_description]
    generated_scores = []
    
    for i, token_id in enumerate(generated_ids):
        token = processor.tokenizer._convert_id_to_token(token_id).replace("▁", " ")
        logits = torch.softmax(forward_logits[i], dim=-1).tolist()
        logits = {t: round(p, 4) for t, p in enumerate(logits) if (p > min_prob or t == token_id)}
        logits = {processor.tokenizer._convert_id_to_token(t).replace("▁", " ").strip(): p for t, p in logits.items()}
        top10_logits = dict(sorted(logits.items(), key=lambda item: item[1], reverse=True)[:10])
        top10_logits[token.strip()] = logits[token.strip()]
        generated_scores.append((token, top10_logits))
    
    generated_text = ''.join([x for x, _ in generated_scores])
    if len(str(generated_scores)) > 32767:
        print("Over excel limit:", image_url)
    print(str(generated_scores))
    return str(generated_scores)


# import pandas as pd 
# df = pd.read_excel(f'/home/student/HallucinationsLLM/result_data1.xlsx', index_col=0)
# df['logits'] = df.apply(lambda row: extract_logits(processor, row['image_link'], row['description']), axis=1)
# df.to_excel('team5_final_data.xlsx')
# image_url = df.iloc[0]['image_link']
# generated_description = df.iloc[0]['description']
# generated_text, generated_scores = extract_logits(processor, image_url, generated_description, min_prob=1e-3)

import pandas as pd 
df = pd.read_excel(f'/home/student/HallucinationsLLM/data/team5_clean_dataset.xlsx', index_col=0)
print(df.iloc[[64]].apply(lambda row: extract_logits(processor, row['image_link'], row['description']), axis=1))