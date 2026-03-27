from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging
)
import pandas as pd
import re
import json
import yaml
from datetime import datetime
from datasets import Dataset
from huggingface_hub import login
from transformers import AutoConfig, AutoModel
from transformers import logging

# Disable warnings from transformers
logging.set_verbosity_error()
import torch
from peft import PeftModel, PeftConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from huggingface_hub import login
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
tqdm.pandas()

# Disable warnings from transformers
logging.set_verbosity_error()

# Hugging Face login
login(token="put your hf token")

# ########## Data Preparation #############
# Assign variables
base_path = "/root/pj_llm/dataset/preprocessed/pkl/"
raw_filename = "wholespine_ori_question.pkl"

# Set models
base_model = "ProbeMedicalYonseiMAILab/medllama3-v20"  # Hugging Face Basic Model
adapter_model_name = "/root/pj_llm/codes/04_hallucination/model_jh"  # Adapter model path

with open(base_path + raw_filename, 'rb') as f:
    raw_dataset = pickle.load(f)

def create_text_column(example):
    return f"### Instruction:\n{example['instruction']}\n\nInput:\n{example['input']}\n\n### Response:\n{example['output']}"

raw_dataset['text'] = raw_dataset.apply(create_text_column, axis=1)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
EOS_TOKEN = tokenizer.eos_token

def prompt_eos(example):
    example['text'] += EOS_TOKEN
    return example

raw_dataset = raw_dataset.apply(prompt_eos, axis=1)

# Split data into train, eval, and test sets
train_dataset, temp_dataset = train_test_split(raw_dataset, test_size=0.30, stratify=raw_dataset['output'], random_state=42)
eval_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.50, stratify=temp_dataset['output'], random_state=42)

train_dataset.reset_index(drop=True, inplace=True)
eval_dataset.reset_index(drop=True, inplace=True)
test_dataset.reset_index(drop=True, inplace=True)

test_dataset = test_dataset.drop(['instruction', 'text', 'output'], axis=1)

# Load results from CSV files
df_before_pe = pd.read_csv("/root/pj_llm/codes/04_hallucination/before_pe_result/before_pe.csv")
df_after_pe = pd.read_csv("/root/pj_llm/codes/04_hallucination/after_pe_result/after_pe.csv")

def prepare_results(df, test_dataset):
    new_df = pd.concat([df, test_dataset], axis=1)
    new_df['report'] = new_df['input']
    new_df = new_df.drop(['input'], axis=1)
    return new_df

new_df_before = prepare_results(df_before_pe, test_dataset)
new_df_after = prepare_results(df_after_pe, test_dataset)

########## PEFT + BASE_MODEL ##############
# Check GPU architecture
if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

# Set quantization with BitsAndBytesConfig
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1

new_model = PeftModel.from_pretrained(model, adapter_model_name)

# Load YAML file for prompts
# Set paths for YAML files
# YAML 파일 경로
prompt_path = '/root/pj_llm/codes/04_hallucination/prompt/nolabel_confidence_score_calculation.yaml'

# YAML 파일 불러오기
with open(prompt_path, 'r', encoding='utf-8') as f:
    prompts_yaml = yaml.safe_load(f)

# 프롬프트 딕셔너리로 저장 (id를 키로 사용)
prompts = {prompt['id']: prompt['text'] for prompt in prompts_yaml['prompts']}

# 추가된 부분: YAML 파일에서 프롬프트 그룹 나누기
before_pe_prompts = {}
after_pe_prompts = {}
split_index = 1  # YAML 파일에서 분리할 인덱스 지정

# 분리 기준에 따라 프롬프트를 분리
for idx, prompt in enumerate(prompts_yaml['prompts']):
    if idx < split_index:
        before_pe_prompts[prompt['id']] = prompt['text']
    else:
        after_pe_prompts[prompt['id']] = prompt['text']

def generate_responses(prompt_text, report):
    return prompt_text.format(report=report)

def normalize_confidence(confidence_dict):
    total = sum(confidence_dict.values())
    if total > 0:
        return {label: conf / total for label, conf in confidence_dict.items()}
    return confidence_dict

def extract_confidences(response_text, labels):
    confidence_dict = {label: 0.0 for label in labels}
    
    # 정규 표현식 패턴: 라벨 뒤에 오는 점수나 퍼센트를 정확히 추출
    pattern = re.compile(
        r'(?P<label>' + '|'.join(re.escape(label) for label in labels) + r')\s*[:\s/]*\s*(?P<confidence>[\d.]+)?%?'
    )
    
    matches = pattern.finditer(response_text)
    
    found_labels = set()
    for match in matches:
        label = match.group('label')
        conf = match.group('confidence')
        
        if conf:
            try:
                confidence = float(conf) / 100.0  # 퍼센트 값을 소수로 변환
            except ValueError:
                confidence = 0.0
        else:
            confidence = 0.0
        
        # 라벨이 confidence_dict에 존재할 때
        if label in confidence_dict:
            confidence_dict[label] = confidence
            found_labels.add(label)
    
    # 모든 confidence 값이 0.0인 경우
    if all(value == 0.0 for value in confidence_dict.values()):
        # response_text에서 라벨을 나열하여 가장 첫 번째 라벨을 찾기
        for label in labels:
            if re.search(r'\b{}\b'.format(re.escape(label)), response_text):
                confidence_dict = {label: 1.0 if label == label else 0.0 for label in labels}
                break

    return confidence_dict

def generate_custom_response(prompt, all_labels, max_new_tokens=60):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response_start = "### response ###"
    response_text = response.split(response_start, 1)[-1].strip() if response_start in response else response.strip()

    print(f"Model response:\n {response_text}")

    confidence_dict = extract_confidences(response_text, all_labels)
    return normalize_confidence(confidence_dict)

def evaluate_model(df, model, tokenizer, prompts, timestamp):
    results = {}
    all_labels = ['no', 'mets', 'progression', 'stable', 'improved', 'romets']

    for prompt_id, prompt_text in prompts.items():
        def generate_response(row):
            report = row['report']
            prompt = generate_responses(prompt_text, report)
            response = generate_custom_response(prompt, all_labels)
            response['predicted_label'] = row['predicted']
            response['true_label'] = row['correct']
            return response

        print(f"Generating responses for prompt_id: {prompt_id}...")
        result = df.progress_apply(generate_response, axis=1)
        result_df = pd.DataFrame(result.tolist())

        y_true = result_df['true_label']
        y_pred = result_df.drop(columns=['predicted_label', 'true_label'])

        lb = LabelBinarizer()
        lb.fit(y_true.unique())
        y_true_binary = lb.transform(y_true)
        y_pred_binary = np.array([list(pred.values()) for pred in y_pred.to_dict('records')])

        auroc_values = {}
        for i, label in tqdm(enumerate(y_pred.columns), total=len(y_pred.columns), desc=f"AUROC calculation for {prompt_id}"):
            if i < y_true_binary.shape[1] and i < y_pred_binary.shape[1]:
                try:
                    auroc = roc_auc_score(y_true_binary[:, i], y_pred_binary[:, i])
                    auroc_values[label] = auroc
                except ValueError:
                    auroc_values[label] = np.nan
            else:
                auroc_values[label] = np.nan
                print(f"Warning: Index {i} is out of bounds for AUROC calculation")

        average_auroc = np.nanmean(list(auroc_values.values()))

        print(f"AUROC results for {prompt_id}:")
        for label, auroc in auroc_values.items():
            print(f"AUROC for {label}: {auroc:.4f}")

        print(f"Average AUROC: {average_auroc:.4f}")

        result_df["average_auroc"] = average_auroc
        for label, auroc in auroc_values.items():
            result_df[f"auroc_{label}"] = auroc

        metrics_save_path = '/root/pj_llm/codes/04_hallucination/auroc_and_normalized_confidence_score_result/'
        os.makedirs(metrics_save_path, exist_ok=True)  # Ensure directory exists
        combined_csv_path = os.path.join(metrics_save_path, f'combined_results_{prompt_id}_{timestamp}.csv')
        result_df.to_csv(combined_csv_path, index=False)

# Example execution
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Before PE evaluation
evaluate_model(new_df_before, new_model, tokenizer, before_pe_prompts, timestamp)

# After PE evaluation
evaluate_model(new_df_after, new_model, tokenizer, after_pe_prompts, timestamp)
