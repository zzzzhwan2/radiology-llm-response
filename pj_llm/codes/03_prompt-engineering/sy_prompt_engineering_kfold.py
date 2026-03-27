import os
import re
import pickle
from datasets import Dataset
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

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

############################################
# Load dataset
base_path = "./pkl/"
raw_filename = "wholespine_ori_question.pkl"

with open(base_path + raw_filename, 'rb') as f:
    raw_dataset = pickle.load(f)

def create_text_column(example):
    text = f"{example['output']}"
    return text

raw_dataset['text'] = raw_dataset.apply(create_text_column, axis=1)

############################################
# Set models
huggingface_hub.login(token="put your hf token", add_to_git_credential=True)
base_model = "ProbeMedicalYonseiMAILab/medllama3-v20" 
adapter_model_name = "./modelfile/models/sy_original_7_len1024_4bit_fullfinetuning/model"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True) 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
EOS_TOKEN = tokenizer.eos_token

def prompt_eos(example):
    example['text'] = example['text'] + EOS_TOKEN
    return example

raw_dataset = raw_dataset.apply(prompt_eos, axis=1)

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
def generate_custom_response(prompt, max_new_tokens=60):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move input tensors to the same device as the model (usually CUDA)
    inputs = {key: value.to(new_model.device) for key, value in inputs.items()}
    
    outputs = new_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.00001)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Convert the response to lowercase
    response = response.lower()
    
    return response

# Evaluate the model using each prompt type
def evaluate_model(test_dataset, new_model, tokenizer, prompt_text, prompt_id, timestamp):
    def generate_response(batch):
        prompts = [prompt_text.format(report=input_text) for input_text in batch['input']]
        generated_texts = [generate_custom_response(prompt, max_new_tokens=200) for prompt in prompts]
        return {'predicted': generated_texts}
    
    # Apply the generate_response function to each batch in the test dataset
    test_dataset = test_dataset.map(generate_response, batched=True, batch_size=1, desc="Processing Dataset")

    correct = test_dataset['output']
    predicted = test_dataset['predicted']
    
    # Regular expression pattern to find relevant labels
    filtered = []
    label_pattern = re.compile(r'\b(?:no|mets|progression|stable|improved|romets)\b', re.IGNORECASE)
    
    for c, p in zip(correct, predicted):
        lines = p.splitlines()
        final_label = None
        probabilities = {}
    
        for line in lines:
            if line.startswith("final label:"):
                final_label = line.split(":", 1)[1].strip()
            else:
                key_value = line.split(":")
                if len(key_value) == 2:
                    try:
                        label, prob = key_value
                        probabilities[label.strip()] = float(prob.strip())
                    except ValueError:
                        pass  # Handle cases where probability conversion fails
    
        if final_label is None:
            # If no 'final label:' is found, process the output section to extract labels
            response_start = "### output ###"
            output_section = p.split(response_start)[-1].strip()
            output_lines = output_section.splitlines()[:2]  # Get the first 2 lines after '### output ###'
            output_text = " ".join(output_lines)
    
            # Use regex to find the label in the first 2 lines after '### output ###'
            match = label_pattern.search(output_text)
            if match:
                final_label = match.group(0).lower()
    
        if final_label is None and probabilities:
            # If no label was extracted, fall back to the label with the highest probability
            final_label = max(probabilities, key=probabilities.get)

        # Append the correct label and the final label determined by the above logic
        filtered.append((c, final_label))

    filtered_df = pd.DataFrame(filtered, columns=['correct', 'predicted'])

    # Encode the labels to numerical values
    le = LabelEncoder()
    le.fit([c for c, _ in filtered] + [p for _, p in filtered])
    true_labels = le.transform([c for c, _ in filtered])
    predicted_labels = le.transform([p for _, p in filtered])

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=1)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    # Print the metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # Save the results
    result_save_path = './pj_llm/results/'
    results_savefolder = os.path.join(result_save_path, timestamp)
    if not os.path.exists(results_savefolder):
        os.makedirs(results_savefolder)

    # Save filtered predictions with raw predicted text
    filtered_and_raw_df = filtered_df.copy()
    filtered_and_raw_df['predicted_raw'] = predicted
    filtered_and_raw_df_csv_path = os.path.join(results_savefolder, f'{prompt_id}_{timestamp}_llms_output.csv')
    filtered_and_raw_df.to_csv(filtered_and_raw_df_csv_path, index=False)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_csv_path = os.path.join(results_savefolder, f'{prompt_id}_{timestamp}_metrics_results.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    return metrics

############################################
# Load and apply prompts
prompt_path = './codes/promt_engineering/prompts/sy_best_and_worst_modified.yaml'
with open(prompt_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

prompts = data.get('prompts', [])

############################################
# K-Fold Cross-Validation
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# List to store metrics for each fold
all_metrics = []

# Perform k-fold cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(raw_dataset, raw_dataset['output'])):
    print(f"Starting fold {fold + 1}/{k_folds}...")

    # Split the dataset into training and testing sets for this fold
    train_data = raw_dataset.iloc[train_index].reset_index(drop=True)
    test_data = raw_dataset.iloc[test_index].reset_index(drop=True)
    
    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)

    for prompt in prompts:
        prompt_id = prompt['id']
        print('=====')
        print(f"Fold {fold + 1}, Prompt ID: {prompt_id}")
        print('=====')
        prompt_text = prompt['text']
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        
        metrics = evaluate_model(test_dataset, new_model, tokenizer, prompt_text, prompt_id, timestamp)
        all_metrics.append(metrics)

# Average the metrics across all folds
average_metrics = {
    'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
    'precision': np.mean([m['precision'] for m in all_metrics]),
    'recall': np.mean([m['recall'] for m in all_metrics]),
    'f1_score': np.mean([m['f1_score'] for m in all_metrics])
}

# Print the averaged metrics
print("\n=== Final Averaged Metrics Across All Folds ===")
print(f'Accuracy: {average_metrics["accuracy"]:.4f}')
print(f'Precision: {average_metrics["precision"]:.4f}')
print(f'Recall: {average_metrics["recall"]:.4f}')
print(f'F1 Score: {average_metrics["f1_score"]:.4f}')

# Save final averaged metrics to CSV
final_metrics_df = pd.DataFrame([average_metrics])
final_metrics_csv_path = os.path.join(result_save_path, f'final_averaged_metrics_{timestamp}.csv')
final_metrics_df.to_csv(final_metrics_csv_path, index=False)