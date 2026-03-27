import os
import re
import pickle
from datasets import Dataset
import pandas as pd
import numpy as np
import random
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler, autocast

import huggingface_hub
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
    EarlyStoppingCallback,
)
from peft import LoraConfig, PeftModel, PeftConfig
from trl import SFTTrainer, SFTConfig
import yaml

# #### seed 고정 ####
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)


############################################
# Load dataset


base_path = "/root/pkl/"
raw_filename = "wholespine_withdescription.pkl"
# train_filename = "wholespine_train.pkl"
# eval_filename = "wholespine_eval.pkl"
# test_filename = "wholespine_test.pkl"

with open(base_path + raw_filename, 'rb') as f:
    raw_dataset = pickle.load(f)

# with open(os.path.join(base_path, train_filename), 'rb') as f:
#     train_dataset = pickle.load(f)

# with open(os.path.join(base_path, eval_filename), 'rb') as f:
#     eval_dataset = pickle.load(f)

# with open(os.path.join(base_path, test_filename), 'rb') as f:
#     test_dataset = pickle.load(f)

def create_text_column(example):
    text = f"### Instruction:\n{example['instruction']}\n\nInput:\n{example['input']}\n\n### Response:\n{example['output']}"
    return text

raw_dataset['text'] = raw_dataset.apply(create_text_column, axis=1)

# train_dataset['text'] = train_dataset.apply(create_text_column, axis=1)
# eval_dataset['text'] = eval_dataset.apply(create_text_column, axis=1)
# test_dataset['text'] = test_dataset.apply(create_text_column, axis=1)

############################################
# Set models
huggingface_hub.login(token="put your hf token", add_to_git_credential=True)
base_model = "ProbeMedicalYonseiMAILab/medllama3-v20" 
adapter_model_name = "/root/modelfile/models/sy_original_7_len1024_4bit_fullfinetuning/model"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True) 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
EOS_TOKEN = tokenizer.eos_token

def prompt_eos(example):
    example['text'] = example['text'] + EOS_TOKEN
    return example

raw_dataset = raw_dataset.apply(prompt_eos, axis=1)

# Divide into train, eval and test
'''
The Ratio of
train dataset: 70%
eval dataset:  15%
test dataset:  15%
'''
train_dataset, temp_dataset = train_test_split(raw_dataset, test_size=0.30, stratify=raw_dataset['output'], random_state=42)
eval_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.50, stratify=temp_dataset['output'], random_state=42)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_dataset)
eval_dataset = Dataset.from_pandas(eval_dataset)
test_dataset = Dataset.from_pandas(test_dataset)
print(f"Number of entries in train_dataset: {len(train_dataset)}")
print(f"Number of entries in eval_dataset: {len(eval_dataset)}")
print(f"Number of entries in test_dataset: {len(test_dataset)}")

# train_dataset = train_dataset.apply(prompt_eos, axis=1)
# eval_dataset = eval_dataset.apply(prompt_eos, axis=1)
# test_dataset = test_dataset.apply(prompt_eos, axis=1)

# train_dataset = Dataset.from_pandas(train_dataset)
# eval_dataset = Dataset.from_pandas(eval_dataset)
# test_dataset = Dataset.from_pandas(test_dataset)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

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
model.eval()

new_model = PeftModel.from_pretrained(model, adapter_model_name)

############################################
generator = pipeline("text-generation", model=new_model, tokenizer=tokenizer, temperature=0.01)

# Evaluate the model using each prompt type
def generate_responses(prompt_type, report):
    return prompt_type.format(report=report)

def evaluate_model(test_dataset, new_model, tokenizer, prompt_text, prompt_id, timestamp):
    def generate_response(batch):
        prompts = [generate_responses(prompt_text, input_text) for input_text in batch['input']]
        responses = generator(prompts, max_new_tokens=10, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        generated_texts = [response[0]['generated_text'].replace(prompt, "").strip() for response, prompt in zip(responses, prompts)]
        return {'predicted': generated_texts}

    test_dataset = test_dataset.map(generate_response, batched=True, batch_size=1, desc="Processing Dataset")

    correct = test_dataset['output']
    predicted = test_dataset['predicted']
    
    pattern = re.compile(r'\b(?:no|mets|progression|stable|improved|romets)\b')
    
    filtered = [(c, pattern.search(p).group(0) if pattern.search(p) else p) for c, p in zip(correct, predicted)]
    print('Was filtering successful? - ', 'YES' if len(filtered) == len(correct) else 'NO')
    
    metrics_save_path = '/root/codes/promt_engineering/results_prompt/sy/'
    filtered_df = pd.DataFrame(filtered, columns=['correct', 'predicted'])
    results_csv_path = os.path.join(metrics_save_path, f'results_{prompt_id}_{timestamp}.csv')
    filtered_df.to_csv(results_csv_path, index=False)

    le = LabelEncoder()
    le.fit([c for c, _ in filtered] + [p for _, p in filtered])
    true_labels = le.transform([c for c, _ in filtered])
    predicted_labels = le.transform([p for _, p in filtered])

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    metrics_df = pd.DataFrame([metrics])
    # metrics_csv_path = os.path.join(metrics_save_path, f'metrics_{prompt_id}_{timestamp}.csv')
    # metrics_df.to_csv(metrics_csv_path, index=False)

    combined_df = pd.concat([filtered_df, metrics_df], axis=1)
    combined_csv_path = os.path.join(metrics_save_path, f'combined_results_{prompt_id}_{timestamp}.csv')
    combined_df.to_csv(combined_csv_path, index=False)

############################################
# Load and apply prompts
prompt_path = '/root/codes/promt_engineering/prompts/sy_new_test_prompt_by_harin.yaml'
with open(prompt_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

prompts = data.get('prompts', [])
# print(prompts)
for prompt in prompts:
    prompt_id = prompt['id']
    print('=====')
    print(prompt_id)
    print('=====')
    prompt_text = prompt['text']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluate_model(test_dataset, new_model, tokenizer, prompt_text, prompt_id, timestamp)
