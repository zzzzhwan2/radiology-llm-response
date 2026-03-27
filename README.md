# 🏥 PHAROS: Prompt Hallucination Avoidance and Response Optimization in Studies of Radiology with LLMs

**연세대학교 세브란스 병원의 영상검사기록지(Radiology Reports)** 를 활용하여, 거대 언어 모델(LLM)이 암 환자의 치료 반응(호전, 전이, 악화 등)을 정확하게 분류하도록 개발한 의료 도메인 특화 LLM 시스템입니다.

🏆 **연세 디지털 헬스케어 사이버보안 경진대회 장려상 수상작** (발주: 연세대학교 의과대학 영상의학과)

## 📝 프로젝트 배경 및 목적
* **기존 수동 판독의 한계**: 방사선 전문의가 직접 영상검사기록지를 검토하고 치료 반응을 판독하는 과정은 많은 시간이 소요되며 주관적 편차가 발생할 수 있습니다.
* **의료 LLM의 환각(Hallucination) 리스크**: 범용 LLM을 의료 데이터에 적용할 경우, 그럴듯하지만 사실과 다른 임상 정보를 생성하는 환각 현상이 발생해 치명적인 임상 의사결정 오류를 초래할 수 있습니다.
* **목표**: 
  1. 의료 특화 사전학습 모델을 파인튜닝(Fine-tuning)하여 암 치료 반응 분류에 특화된 LLM 구축
  2. 임상의 자문을 반영한 진단 프로세스 기반 프롬프트 엔지니어링(Prompt Engineering) 적용
  3. 환각 현상을 추적하고 완화하기 위한 새로운 평가지표(Likelihood 기반 AUROC) 및 검증 파이프라인 제시

## 📊 데이터셋 (Dataset)
* **출처**: 연세대학교 세브란스 영상의학과 비식별화 영상검사기록지 700건 (2002년~2024년)
* **분류 라벨 (6 Classes)**: 
  * `improved` (호전), `mets` (전이), `no` (이상없음), `progression` (악화), `romets` (전이 의심/추가검사 필요), `stable` (변화없음)

## 🏗️ 핵심 방법론 (Methodology)

### 1. Model Fine-Tuning (QLoRA)
제한된 GPU 환경(RTX 3090 1대)에서 효율적인 학습을 위해 QLoRA 기법을 사용했습니다.
* **모델 선정**: 다양한 모델(`Meerkat-7B`, `radiology-llama2 7B`, `Llama-3.1 8B` 등) 비교 결과, 가장 성능이 우수한 의료 도메인 특화 언어모델인 **`Medllama3-v20 8B`** 채택.
* **하이퍼파라미터 최적화**: `Max sequence length 1024`, `4bit 양자화`, `Full epoch (30)` 설정으로 파인튜닝 최적화 달성.

### 2. Medical Prompt Engineering
단순 지시문을 넘어, 전문의의 실제 진단 논리를 LLM에 이식하기 위해 7단계의 프롬프트 실험을 수행했습니다.
* **적용 기법**: 진단 프로세스 명시(Diagnosis Process) + 단계적 추론(Chain of Thoughts; CoT) + 특정 라벨(Rule-Out) 예외 처리 규칙 삽입
* **최적 프롬프트**: `Diagnosis process + CoT + Rule-out` 조합이 가장 높은 정확도를 기록.

### 3. Hallucination Analysis & Mitigation
모델의 분류 결과에 대한 임상적 신뢰도를 확보하기 위해 3단계 환각 분석 파이프라인을 구축했습니다.
* **Likelihood 기반 AUROC**: LLM의 출력 확률(Probability) 대신 내부 정답 신뢰도(Likelihood)를 추출하여 예측의 확신도를 측정하는 새로운 평가지표 도입.
* **정성적 원인 분석 (Qualitative Approach)**: 모델이 오답을 냈을 때 추론 과정을 되묻는 추가 프롬프트를 통해, `romets(전이 의심)`를 `improved(호전)`로 오분류하는 등의 구체적인 환각 유발 패턴을 식별.

## 🚀 주요 성과 (Key Results)
* **최고 분류 성능 달성 (F1-Score 0.95)**: 최적화된 파인튜닝 모델에 전문의 자문 기반의 프롬프트를 적용하여, Accuracy, Precision, Recall, F1-Score 등 **모든 핵심 지표에서 0.95**를 달성했습니다. (Base Model 대비 약 21% 향상)
* **환각 제어 능력 입증**: 프롬프트 엔지니어링 전후를 비교했을 때, 모델의 정답 신뢰도를 나타내는 AUROC 지표가 0.53에서 0.63으로 향상되어 보다 확신 있는 정답 생성이 가능해졌습니다.
* **의료 인공지능 실증 기반 마련**: 모델의 단순 예측을 넘어, 어떤 키워드에서 오판이 발생했는지(예: "R/O" 키워드 인식 오류)를 추적할 수 있는 XAI(설명 가능한 AI) 환경을 성공적으로 구축했습니다.

---

## 📂 폴더 구조 (Directory Structure)
본 리포지토리는 데이터 전처리부터 환각 분석까지 전체 파이프라인의 코드를 포함하고 있습니다.

```text
.
├── 01_preprocessing/            # 데이터 전처리 및 탐색적 데이터 분석(EDA)
│   ├── split_dataset_with_eval_sy.py
│   └── wholespine_EDA.ipynb
├── 02_fine-tuning/              # 사전학습 모델 로드 및 QLoRA 파인튜닝 
│   └── finetuned_model_sy_cash_discard_and_seed42.py
├── 03_prompt-engineering/       # 7가지 프롬프트 엔지니어링 실험 및 평가 (K-fold)
│   ├── prompts/                 # yaml 형태의 프롬프트 템플릿 모음
│   ├── prompt_engineering_jihwan.py
│   └── sy_prompt_engineering_kfold.py
└── 04_hallucination/            # 환각 분석, Confidence Score 추출 및 AUROC 계산
    ├── linguistic_calibration_code/
    └── prompt/                  # 환각 분석용 설명(Explanation) 요청 프롬프트
