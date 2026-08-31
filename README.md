# Hate Speech Toxicity Detector

## Final-year project scope

**Project Name:** Hate speech toxicity detector  
**Domain:** NLP

**Project description preserved from the project proposal:**  
Fine-tune DistilBERT / RoBERTa on Twitter Hate Speech data to flag toxic posts.

**Required outputs:**  
- toxicity
- insult
- threat
- identity attack
- attention-weight explanations
- bulk CSV evaluation
- ethical NLP classification
- multi-label modelling
- bias analysis in hate-speech datasets

## Real-world application

The system is designed as a moderation-assistance service for a discussion/forum platform. It accepts a comment/post, predicts the four requested categories independently, shows confidence, highlights attention-weighted tokens, and supports CSV batch evaluation. It should be treated as a decision-support tool, not an automatic punishment system.

## Important data rule

The project specification above is kept unchanged. The code does not invent labels. Your actual training file must contain the four target labels or a documented mapping from the labels in your source dataset.

Recommended canonical CSV schema:

```text
text,toxicity,insult,threat,identity_attack,source
```

The four target columns are binary 0/1 values. `source` is `twitter`, `youtube`, or another documented source.

If your original CSV uses different column names, edit only `training/config.py` rather than rewriting the project.

## Architecture

```text
Twitter/X authorized export/API ----\
                                      \
YouTube authorized API/export --------> Raw CSVs
                                        |
                                        v
                                Data validation + merge
                                        |
                                        v
                              Train/validation/test split
                                        |
                                        v
                           RoBERTa/DistilBERT fine-tuning
                                        |
                                        v
                              Saved model + tokenizer
                                        |
                                        v
                  Flask backend <------ inference service
                       |                  |
                       |                  +--> multi-label prediction
                       |                  +--> attention evidence
                       |                  +--> CSV bulk evaluation
                       |
                       +--> REST API
                       |
                       v
                 HTML/CSS/JS frontend
                       |
                       v
                   Deployment
```

## Training recommendation

Use **Google Colab GPU** for model fine-tuning. Hugging Face's Trainer API handles batching, padding, backpropagation, evaluation, checkpointing and related training tasks. See the official fine-tuning documentation:
https://huggingface.co/docs/transformers/en/training

Start with:

```text
MODEL_NAME=distilbert-base-uncased
```

for faster experiments, then compare against:

```text
MODEL_NAME=roberta-base
```

for the final experimental comparison.

## Folder structure

```text
HateSpeechToxicityDetector/
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── services/
│   │   ├── inference.py
│   │   └── explanations.py
│   └── utils/
│       └── validation.py
├── frontend/
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── collection/
│   ├── youtube_collector.py
│   └── twitter_export_normalizer.py
├── training/
│   ├── config.py
│   ├── prepare_dataset.py
│   ├── train_multilabel.py
│   ├── evaluate_model.py
│   └── colab_setup.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── deployment/
│   ├── Dockerfile
│   ├── gunicorn.conf.py
│   └── render.yaml
├── requirements.txt
└── .env.example
```

## Local inference setup

Use Python 3.11.

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set:

```text
MODEL_PATH=models/final_model
```

Run:

```bash
python backend/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Google Colab workflow

1. Open Google Colab.
2. Select Runtime -> Change runtime type -> GPU.
3. Upload the project ZIP or clone your repository.
4. Put your training CSV at `data/raw/all_comments.csv`.
5. Run `training/colab_setup.py`.
6. Run `training/prepare_dataset.py`.
7. Run `training/train_multilabel.py`.
8. Run `training/evaluate_model.py`.
9. Download `models/final_model/`.
10. Put that folder into the deployed backend.

## Data collection

Do not scrape or collect data in ways that violate the platform's terms or privacy rules. Prefer:
- your own authorized exports,
- datasets whose license permits research use,
- official APIs where you have the required access,
- YouTube comments you are authorized to process.

The collector scripts in this project only normalize data into the project's canonical schema. They do not bypass authentication, rate limits, age gates, or platform restrictions.

## Evaluation

Report at least:

- micro F1
- macro F1
- weighted F1
- per-label precision/recall/F1
- per-label ROC-AUC when meaningful
- confusion matrix for each label
- threshold used for each label
- source-wise performance: Twitter vs YouTube
- false-positive and false-negative examples
- class distribution
- bias/slice analysis

Accuracy alone is not enough for multi-label hate-speech classification.

## Ethical NLP component

The dashboard includes a bias-analysis endpoint. Compare performance across available dataset slices such as source, language, or another non-sensitive dataset field you are legitimately allowed to use. Do not infer sensitive personal attributes about individual users.

## Production note

The final deployed system should be presented as **moderation assistance**. A high-confidence model prediction should trigger human review or an existing moderation workflow rather than automatically deciding a person's account status.

## API endpoints

- `GET /api/health`
- `POST /api/predict`
- `POST /api/bulk`
- `GET /`

`POST /api/predict` JSON:

```json
{
  "text": "example comment"
}
```

Response:

```json
{
  "text": "example comment",
  "labels": {
    "toxicity": {"flagged": true, "score": 0.91},
    "insult": {"flagged": true, "score": 0.83},
    "threat": {"flagged": false, "score": 0.04},
    "identity_attack": {"flagged": false, "score": 0.12}
  },
  "attention": [
    {"token": "example", "weight": 0.10}
  ]
}
```

## What to explain during viva

### Algorithms
1. Transformer-based contextual language representation.
2. RoBERTa/DistilBERT fine-tuning.
3. Multi-label classification with four independent sigmoid outputs.
4. Binary cross-entropy style multi-label loss.
5. Attention-based token importance visualization.
6. Precision, recall and F1 evaluation.
7. Threshold-based decision making.

### Why multi-label?
One comment can simultaneously be toxic, insulting and an identity attack. The four labels are therefore not mutually exclusive.

### Why Google Colab?
GPU acceleration makes transformer fine-tuning practical for a student project and keeps the deployment machine separate from the training environment.

### What makes it real-world?
It is not only a one-text prediction demo. It provides:
- a REST API,
- web moderation UI,
- batch CSV processing,
- source tracking,
- explanations,
- evaluation reports,
- bias analysis,
- deployment-ready backend.

