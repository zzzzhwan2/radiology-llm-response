import os
import re
import pickle
from datasets import Dataset
import pandas as pd
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler, autocast

import random
import numpy as np

#### seed 고정 ####
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
###################

#캐시 제거
import torch
torch.cuda.empty_cache()


import huggingface_hub
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    EarlyStoppingCallback,  # Importing EarlyStoppingCallback
)

from transformers import TrainerCallback #추가

from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model
)
from trl import SFTTrainer, SFTConfig

# Login to Hugging Face Hub
huggingface_hub.login(token="put your hf token", add_to_git_credential=True)

# Assign variables
base_path = "/root/pkl/"
# raw_filename = "wholespine_ori_question.pkl"

# Set models
base_model = "ProbeMedicalYonseiMAILab/medllama3-v20"           # Hugging Face Basic Model
new_model = "ProbeMedicalYonseiMAILab/medllama3-v20-finetuned"  # Fine tuning Model
raw_dataset = pd.read_csv('/root/codes/error_anlysis/raw_dataset_clear.csv')

# with open(base_path + raw_filename, 'rb') as f:
#     raw_dataset = pickle.load(f)

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
# train_dataset, temp_dataset = train_test_split(raw_dataset, test_size=0.2, stratify=raw_dataset['output'], random_state=42)
# eval_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, stratify=temp_dataset['output'], random_state=42)
# Reset index
train_dataset.reset_index(drop=True, inplace=True)
eval_dataset.reset_index(drop=True, inplace=True)
test_dataset.reset_index(drop=True, inplace=True)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_dataset)
eval_dataset = Dataset.from_pandas(eval_dataset)
test_dataset = Dataset.from_pandas(test_dataset)
print(f"Number of entries in train_dataset: {len(train_dataset)}")
print(f"Number of entries in eval_dataset: {len(eval_dataset)}")
print(f"Number of entries in test_dataset: {len(test_dataset)}")

# Ensure that we are using GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Check GPU architecture and set attention implementation and dtype accordingly
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

# quant_config = BitsAndBytesConfig(
#     load_in_8bit=True,  # 8비트 양자화 설정
#     bnb_8bit_quant_type="int8",  # 양자화 타입 설정 (8비트 정수)
#     bnb_8bit_compute_dtype=torch.float32,  # 연산에 사용할 데이터 타입
#     bnb_8bit_use_double_quant=False,  # 추가적인 양자화 사용 여부
# )

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# PEFT configuration
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training configuration
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=30,               # 30 고정
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,     # 1 고정
    optim="paged_adamw_32bit",
    #eval_steps=25,                    # Frequency of evaluation
    #save_steps=25,
    logging_steps=50,
    eval_strategy="epoch",            # Add evaluation strategy
    save_strategy="epoch",
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    tf32=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    load_best_model_at_end=True,      # Load the best model at the end of training
)

# 병렬처리 with DataParallel
if torch.cuda.device_count() > 1:
    model = DataParallel(model)

# 학습 중간에 에포크 끝날 때 캐시를 비우는 콜백 클래스 정의
class ClearCacheCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cache cleared at the end of epoch")
        return control


# Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=600,
    tokenizer=tokenizer,
    args=training_params,
    packing=False ,
    callbacks=[ClearCacheCallback()]  # , EarlyStoppingCallback(early_stopping_patience=5) Add clearing cash callback and early stopping callback
)
trainer.train()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Ensure the directory exists before saving
# result_save_path = '/root/modelfile/models/sy_original_7_len600_4bit-fullfinetuning/'
result_save_path = 'root/codes/error_anlysis/results/'
os.makedirs(f'{result_save_path}', exist_ok=True)

# Save the model weights and tokenizer
model.save_pretrained(result_save_path + "model")
tokenizer.save_pretrained(result_save_path + "tokenizer")

# Define the text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

key = """Based on the report below, classify a cancer treatment response into one of six labels: improved, mets, no, progression, romets, stable.\n\n- Description of six labels\n    - no\n        - This label indicates the absence of any detectable abnormality or condition in the imaging study.\n        - For example, 'no metastasis' means that the imaging did not reveal any evidence of cancer spreading to other parts of the body.\n    - mets\n        - Metastasis refers to the spread of cancer cells from the primary site to other parts of the body.\n        - This label is used when imaging shows evidence of such spread, which is critical for staging and treatment planning.\n    - progression\n        - The term 'progression' is used when there is evidence that the disease is worsening or advancing.\n        - In the context of cancer, this means the tumor has grown in size or spread further since the last imaging study.\n    - stable\n        - 'Stable' indicates that there has been no significant change in the condition or findings since the previous imaging study.\n        - This means that the disease has neither progressed nor improved, remaining unchanged.\n    - improved\n        - This label signifies that there has been a positive change or reduction in the severity of the disease or abnormality.\n        - For instance, in cancer, it means the tumor size has decreased or the extent of the disease has lessened compared to prior imaging.\n    - romets\n        - 'Rule out metastasis' is used when further investigation is needed to determine if metastasis is present.\n        - It indicates that additional diagnostic tests are required to confirm or exclude the presence of metastatic disease."""
# key 승현ver
# key = """Classify the radiology report into one of six categories: "stable", "romets", "progression", "no", "mets", "improved".\n\n- Description of six labels\n    - no\n        - This label indicates the absence of any detectable abnormality or condition in the imaging study.\n        - For example, 'no metastasis' means that the imaging did not reveal any evidence of cancer spreading to other parts of the body.\n    - mets\n        - Metastasis refers to the spread of cancer cells from the primary site to other parts of the body.\n        - This label is used when imaging shows evidence of such spread, which is critical for staging and treatment planning.\n    - progression\n        - The term 'progression' is used when there is evidence that the disease is worsening or advancing.\n        - In the context of cancer, this means the tumor has grown in size or spread further since the last imaging study.\n    - stable\n        - 'Stable' indicates that there has been no significant change in the condition or findings since the previous imaging study.\n        - This means that the disease has neither progressed nor improved, remaining unchanged.\n    - improved\n        - This label signifies that there has been a positive change or reduction in the severity of the disease or abnormality.\n        - For instance, in cancer, it means the tumor size has decreased or the extent of the disease has lessened compared to prior imaging.\n    - romets\n        - 'Rule out metastasis' is used when further investigation is needed to determine if metastasis is present.\n        - It indicates that additional diagnostic tests are required to confirm or exclude the presence of metastatic disease."""

def evaluate_model(test_dataset, model, tokenizer, timestamp):
    
    def generate_response(batch):
        prompts = [f"### Instruction:\n{key}\n\n### Input:\n{input_text}\n\n### Response:" for input_text in batch['input']]
        responses = generator(prompts, max_new_tokens=10, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        generated_texts = [response[0]['generated_text'].replace(prompt, "").strip() for response, prompt in zip(responses, prompts)]
        return {'predicted': generated_texts}

    test_dataset = test_dataset.map(generate_response, batched=True, batch_size=1)  # Adjust batch_size as needed

    correct = test_dataset['output']
    predicted = test_dataset['predicted']

    # Define the regular expression pattern to match the specified labels
    pattern = re.compile(r'\b(?:no|mets|progression|stable|improved|romets)\b')  # Regular expression pattern
    
    # Filter the predicted labels using the regex pattern
    filtered = [(c, pattern.search(p).group(0) if pattern.search(p) else p) for c, p in zip(correct, predicted)]  # Using regex to filter and extract labels
    print('Was filtering successful? - ', 'YES' if len(filtered)==len(correct) else 'NO')
    
    # Ensure the directory for metric exists before saving
    metric_path = f'{result_save_path}/metrics/{timestamp}'
    os.makedirs(metric_path+'/', exist_ok=True)

    # Save filtered correct and predicted labels to a CSV file 
    filtered_df = pd.DataFrame(filtered, columns=['correct', 'predicted'])  # Creating dataframe with filtered data
    results_csv_path = f'{metric_path}/results_{timestamp}.csv'
    filtered_df.to_csv(results_csv_path, index=False)

    # LabelEncoder
    le = LabelEncoder()
    le.fit([c for c, _ in filtered] + [p for _, p in filtered])  # Using filtered values for encoding
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

    # Write metric result to an json file
    metrics_filename = f'{metric_path}/metrics_{timestamp}.json'
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

# Evaluate the model
evaluate_model(test_dataset, model, tokenizer, timestamp)

# ------------------------------- MERGE --------------------------------
# # Ensure the directory exists before saving
# final_model_save_path = f'{result_save_path}/final_model/{timestamp}/'
# os.makedirs(final_model_save_path, exist_ok=True)

# # PEFT model load and merge
# base = AutoModelForCausalLM.from_pretrained(
#      base_model,
#      low_cpu_mem_usage=True,
#      return_dict=True,
#      torch_dtype=torch.float16,
#      device_map="auto")

# finetuned_model_path = f"{result_save_path}/model/"
# fine_tuned_model_loaded = PeftModel.from_pretrained(finetuned_model_path)

# final_model = fine_tuned_model_loaded.merge_and_unload(base_model=base)

# # Save the merged model and tokenizer
# final_model.save_pretrained(final_model_save_path + "final_model")
# tokenizer.save_pretrained(final_model_save_path + "final_tokenizer")

# Reload tokenizer - 필요X
# tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

# Pushing the merged model to Hugging Face Hub - 필요X
#model.push_to_hub(upload_model)
#tokenizer.push_to_hub(upload_model)