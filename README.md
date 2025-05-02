# Reproduction of MTXLSum with NeutralRR Evaluation

This repository contains a Colab notebook that reproduces part of the findings from the paper:

**[Multi-Target Cross-Lingual Summarization: a novel task and a language-neutral approach](https://aclanthology.org/2024.findings-emnlp.755.pdf)**  
_EMNLP 2024 Findings_

## 📌 What’s Reproduced

This notebook focuses on evaluating **NeutralRR**, the proposed language-neutral reranking strategy from the MTXLSum paper. Specifically, it:

- Uses the original **CrossSum** dataset to simulate multi-target summarization.
- Generates summaries using a multilingual model.
- Applies the **NeutralRR** algorithm to select the best candidate summary.
- Evaluates the selected summaries using:
  - **ROUGE-2 (F1)** using `rouge-score`
  - **BLEU** (via NLTK `sentence_bleu`)
  - **BLASER** (approximated as average pairwise semantic similarity using `neutral_rr`)
  - **COMET** (`wmt20-comet-da`)

## 📂 Dataset Used

Dataset: [CrossSum (csebuetnlp/CrossSum)](https://huggingface.co/datasets/csebuetnlp/CrossSum)

- Source: Hugging Face Datasets
- Languages covered: English, French, Spanish, Chinese Simplified, etc.
- Data split used: `test`

## 🧪 Evaluation Metrics

| Metric  | Description                                        |
| ------- | -------------------------------------------------- |
| ROUGE-2 | Measures bigram overlap between prediction and ref |
| BLEU    | Token overlap with smoothing (sentence-level)      |
| BLASER  | Used for semantic similarity amongst the summaries |
| COMET   | Referenced, quality estimation metric              |

## ⚙️ Dependencies

- `datasets`
- `evaluate`
- `rouge-score`
- `unbabel-comet`
- `nltk`
- `torch`
