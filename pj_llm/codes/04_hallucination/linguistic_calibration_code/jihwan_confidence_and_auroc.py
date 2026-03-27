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
# from huggingface_hub import notebook_login
# notebook_login()

# access_token = os.environ["put your hf token"]
# login(token=access_token)


login(token="put your hf token")

# ########## Data Preparation #############
# Assign variables
base_path = "/root/pj_llm/dataset/preprocessed/pkl/"
raw_filename = "wholespine_ori_question.pkl"

# Set models
base_model = "ProbeMedicalYonseiMAILab/medllama3-v20"           # Hugging Face Basic Model
new_model = "ProbeMedicalYonseiMAILab/medllama3-v20-finetuned"  # Fine tuning Model

with open(base_path + raw_filename, 'rb') as f:
    raw_dataset = pickle.load(f)

def create_text_column(example):
    text = f"### Instruction:\n{example['instruction']}\n\nInput:\n{example['input']}\n\n### Response:\n{example['output']}"
    return text

raw_dataset['text'] = raw_dataset.apply(create_text_column, axis=1)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 일관성을 위해 처음부터 설정
EOS_TOKEN = tokenizer.eos_token

def prompt_eos(example):
    example['text'] = example['text'] + EOS_TOKEN
    return example

raw_dataset = raw_dataset.apply(prompt_eos, axis=1)

# Divide into train, eval and test
'''
The Ratio of
train dataset: 70% / 80%
eval dataset:  15% / 10%
test dataset:  15% / 10%
'''
# Split data to train/eval/test
train_dataset, temp_dataset = train_test_split(raw_dataset, test_size=0.30, stratify=raw_dataset['output'], random_state=42)
eval_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.50, stratify=temp_dataset['output'], random_state=42)

# Reset index
train_dataset.reset_index(drop=True, inplace=True)
eval_dataset.reset_index(drop=True, inplace=True)
test_dataset.reset_index(drop=True, inplace=True)

test_dataset = test_dataset.drop(['instruction', 'text', 'output'], axis=1)

# sequence_length 1024, 4bit quantization full-finetuning result data
df_1 = pd.read_csv("/root/pj_llm/codes/04_hallucination/before_pe_result/before_pe.csv") #before pe
new_df = pd.concat([df_1, test_dataset], axis=1)
new_df['report'] = new_df['input']
new_df = new_df.drop(['input'], axis=1)

df_2 = pd.read_csv("/root/pj_llm/codes/04_hallucination/after_pe_result/after_pe.csv") #after pe
new_df2 = pd.concat([df_2, test_dataset], axis=1)
new_df2['report'] = new_df2['input']
new_df2 = new_df2.drop(['input'], axis=1)

########## PEFT + BASE_MODEL ##############
base_model_name = "ProbeMedicalYonseiMAILab/medllama3-v20"
adapter_model_name = "/root/pj_llm/codes/04_hallucination/model_jh"

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
    base_model_name,
    quantization_config=quant_config,
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1

new_model = PeftModel.from_pretrained(model, adapter_model_name) #파라미터 합치는 코드
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True) #토크나이저는 베이스모델것과 동일하므로 상관없음

############################################
# Set paths for YAML files
# YAML 파일 경로
prompt_path = '/root/pj_llm/codes/04_hallucination/prompt/copy2_confidence_score_calculation.yaml'

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

def generate_responses(prompt_text, report, y_i):
    return prompt_text.format(report=report, y_i=y_i)

def normalize_confidence(confidence_dict):
    total = sum(confidence_dict.values())
    if total > 0:
        return {label: conf / total for label, conf in confidence_dict.items()}
    return confidence_dict


def extract_confidences(response_text, all_labels):
    """
    Extract confidence scores from the response text.
    
    Args:
    - response_text (str): The text containing confidence scores.
    - all_labels (list): List of all possible labels.
    
    Returns:
    - dict: A dictionary with labels as keys and confidence scores as values.
    """
    # 정규 표현식으로 모든 라벨과 그에 따른 점수를 추출
    pattern = re.compile(r'(\w+)\s*[:,\s]*([\d.]+)')
    matches = pattern.findall(response_text)

    #디버깅 출력
    print(f"Extracted matches: {matches}")

    # 라벨과 점수의 매칭 딕셔너리 생성
    confidence_dict = {}
    for label, conf in matches:
        if label in all_labels:
            # '.'을 포함한 값은 0.0으로 변환
            if conf == '.':
                confidence_dict[label] = 0.0
            else:
                try:
                    confidence_dict[label] = float(conf)
                except ValueError:
                    confidence_dict[label] = 0.0  # 변환 실패 시 기본값 사용
    
    # 모든 라벨에 대해 값이 없는 경우 0으로 설정
    for label in all_labels:
        if label not in confidence_dict:
            confidence_dict[label] = 0.0
    
    return confidence_dict

def generate_custom_response(prompt, all_labels, max_new_tokens=60):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # print(f"Model response: {response}")  # 디버깅을 위해 추가

    # 예상된 형식의 응답 시작점 찾기
    response_start = "### response ###"
    response_text = response.split(response_start)[-1].strip()

    # 텍스트에서 라벨과 confidence 값을 추출하는 정규 표현식 사용
    confidence_dict = extract_confidences(response_text, all_labels)
    
    # confidence_dict에 유효한 값이 없는 경우 기본값 추가
    if not confidence_dict:
        confidence_dict = {label: 0.0 for label in all_labels}
    
    return normalize_confidence(confidence_dict)



def evaluate_model(new_df, new_model, tokenizer, prompts, timestamp):
    results = {}

    # 모든 라벨을 정의합니다.
    all_labels = ['no', 'mets', 'progression', 'stable', 'improved', 'romets']
    
    for prompt_id, prompt_text in prompts.items():
        def generate_response(row):
            y_i = row['predicted']
            report = row['report']
            prompt = generate_responses(prompt_text, report, y_i)
            response = generate_custom_response(prompt, all_labels)
            response['predicted_label'] = y_i
            response['true_label'] = row['correct']
            return response

        # 프롬프트 별 결과 생성
        print(f"Generating responses for prompt_id: {prompt_id}...")
        result = new_df.progress_apply(generate_response, axis=1)
        result_df = pd.DataFrame(result.tolist())
        # print(result_df.head())  # 디버깅을 위해 추가

        # AUROC 계산을 위해 각 라벨에 대한 데이터 프레임 생성
        y_true = result_df['true_label']
        y_pred = result_df.drop(columns=['predicted_label', 'true_label'])

        lb = LabelBinarizer()
        lb.fit(y_true.unique())
        y_true_binary = lb.transform(y_true)
        y_pred_binary = np.array([list(pred.values()) for pred in y_pred.to_dict('records')])

        auroc_values = {}
        for i, label in tqdm(enumerate(y_pred.columns), total=len(y_pred.columns), desc=f"AUROC calculation for {prompt_id}"):
            try:
                auroc = roc_auc_score(y_true_binary[:, i], y_pred_binary[:, i])
                auroc_values[label] = auroc
            except ValueError:
                auroc_values[label] = np.nan

        # AUROC 점수의 평균 계산
        average_auroc = np.nanmean(list(auroc_values.values()))

        print(f"AUROC results for {prompt_id}:")
        for label, auroc in auroc_values.items():
            print(f"AUROC for {label}: {auroc:.4f}")

        print(f"Average AUROC: {average_auroc:.4f}")

        # 평균 AUROC 점수를 데이터프레임에 추가
        result_df["average_auroc"] = average_auroc

        # AUROC 결과를 데이터프레임에 추가
        for label, auroc in auroc_values.items():
            result_df[f"auroc_{label}"] = auroc

        # 결과를 CSV 파일로 저장
        metrics_save_path = '/root/pj_llm/codes/04_hallucination/auroc_and_normalized_confidence_score_result/'
        combined_csv_path = os.path.join(metrics_save_path, f'combined_results_{prompt_id}_{timestamp}.csv')
        result_df.to_csv(combined_csv_path, index=False)

# 실행 예시
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Before PE 평가
evaluate_model(new_df, new_model, tokenizer, before_pe_prompts, timestamp)

# After PE 평가
evaluate_model(new_df2, new_model, tokenizer, after_pe_prompts, timestamp)