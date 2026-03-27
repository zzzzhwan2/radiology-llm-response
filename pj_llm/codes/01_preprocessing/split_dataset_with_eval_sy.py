import pandas as pd
import re
import pickle
from datasets import Dataset
from sklearn.model_selection import train_test_split

# raw data 경로
rawdata_path = "./pj_llm/dataset/rawdata/WholeSpine700.xlsx"
# 데이터 불러오기
df = pd.read_excel(rawdata_path)
# 데이터 복사
df_copy = df.copy()

# 데이터 전처리 -> 빈컬럼 생성 및 컬럼 재정의 (Alpaca style의 데이터셋 생성)
df_copy["input"]=df_copy['Reports']
df_copy['instruction'] = """Based on the report below, classify a cancer treatment response into one of six labels: improved, mets, no, progression, romets, stable.

- Description of six labels
    - 'no'
        - This label indicates the absence of any detectable abnormality or condition in the imaging study.
        - For example, "no metastasis" means that the imaging did not reveal any evidence of cancer spreading to other parts of the body.
    - 'mets'
        - Metastasis refers to the spread of cancer cells from the primary site to other parts of the body. 
        - This label is used when imaging shows evidence of such spread, which is critical for staging and treatment planning.
    - 'progression'
        - The term "progression" is used when there is evidence that the disease is worsening or advancing.
        - In the context of cancer, this means the tumor has grown in size or spread further since the last imaging study.
    - 'stable'
        - "Stable" indicates that there has been no significant change in the condition or findings since the previous imaging study. 
        - This means that the disease has neither progressed nor improved, remaining unchanged.
    - ‘improved'
        - This label signifies that there has been a positive change or reduction in the severity of the disease or abnormality.
        - For instance, in cancer, it means the tumor size has decreased or the extent of the disease has lessened compared to prior imaging .
    - 'romets'
        - "Rule out metastasis" is used when further investigation is needed to determine if metastasis is present.
        - It indicates that additional diagnostic tests are required to confirm or exclude the presence of metastatic disease.
"""
# df_copy['instruction'] = """Classify the radiology report into one of six categories: "stable", "romets", "progression", "no", "mets", "improved".

# - Description of six labels
#     - 'no'
#         - This label indicates the absence of any detectable abnormality or condition in the imaging study.
#         - For example, "no metastasis" means that the imaging did not reveal any evidence of cancer spreading to other parts of the body.
#     - 'mets'
#         - Metastasis refers to the spread of cancer cells from the primary site to other parts of the body. 
#         - This label is used when imaging shows evidence of such spread, which is critical for staging and treatment planning.
#     - 'progression'
#         - The term "progression" is used when there is evidence that the disease is worsening or advancing.
#         - In the context of cancer, this means the tumor has grown in size or spread further since the last imaging study.
#     - 'stable'
#         - "Stable" indicates that there has been no significant change in the condition or findings since the previous imaging study. 
#         - This means that the disease has neither progressed nor improved, remaining unchanged.
#     - ‘improved'
#         - This label signifies that there has been a positive change or reduction in the severity of the disease or abnormality.
#         - For instance, in cancer, it means the tumor size has decreased or the extent of the disease has lessened compared to prior imaging .
#     - 'romets'
#         - "Rule out metastasis" is used when further investigation is needed to determine if metastasis is present.
#         - It indicates that additional diagnostic tests are required to confirm or exclude the presence of metastatic disease.
# """
df_copy['output'] = df_copy['GT_label']

df_copy = df_copy.drop(['Reports', 'GT_label'], axis=1)

# no 라벨 통일
df_copy['output'] = df_copy['output'].apply(lambda x: re.sub(r'[N|n]o\s*', 'no', x))

# 데이터셋 나누기
# train과 test로 분할
#train_data, test_data = train_test_split(df_copy, test_size=0.2, stratify=df_copy['output'], random_state=42)
# train 데이터를 다시 train과 eval로 분할 
#train_data, eval_data = train_test_split(train_data, test_size=0.1, stratify=train_data['output'], random_state=42)
# reset index
#train_data.reset_index(drop=True, inplace=True)
#eval_data.reset_index(drop=True, inplace=True)
#test_data.reset_index(drop=True, inplace=True)

#print(f"Train data size: {len(train_data)}")
#print(f"Validation data size: {len(eval_data)}")
#print(f"Test data size: {len(test_data)}")

# Save dataframe as pickle
base_path = "./pj_llm/dataset/preprocessed/pkl/"
with open(base_path + "wholespine_ori_question.pkl", 'wb') as f:
    pickle.dump(df_copy, f)

# Save dataframe as csv
df_copy.to_csv("./pj_llm/dataset/preprocessed/csv/wholespine_ori_question.csv", index=False)

#with open(base_path + "wholespine_train.pkl", 'wb') as f:
#    pickle.dump(train_data, f)

#with open(base_path + "wholespine_eval.pkl", 'wb') as f:
#    pickle.dump(eval_data, f)

#with open(base_path + "wholespine_test.pkl", 'wb') as f:
#    pickle.dump(test_data, f)

# base_path = "./pj_llm/dataset/preprocessed/pkl/"
# train_filename = "wholespine_train.pkl"
# eval_filename = "wholespine_eval.pkl"
# test_filename = "wholespine_test.pkl"

# with open(base_path + train_filename, 'rb') as f:
#     train_dataset = pickle.load(f)

# with open(base_path + eval_filename, 'rb') as f:
#     eval_dataset = pickle.load(f)
    
# with open(base_path + test_filename, 'rb') as f:
#     test_dataset = pickle.load(f)

# print(len(train_dataset), len(eval_dataset), len(test_dataset), sep="\n")