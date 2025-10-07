# Fine-Tuning Mistral-7B on AG News with LoRA and 4-bit Quantization

## 🧩 Project Overview

This repository demonstrates fine-tuning of the **Mistral-7B** large language model on a subset of the **AG News dataset** using **LoRA (Low-Rank Adaptation)** combined with **4-bit quantization** for efficient training.  
The model classifies news articles into four predefined categories:

- 🌍 **World**
- ⚽ **Sports**
- 💼 **Business**
- 💻 **Sci/Tech**

The entire fine-tuning pipeline is implemented in the following Jupyter Notebook:  
🔗 [LLM_TUNING_20251005_FINAL.ipynb](https://github.com/subranshu-p/genai/blob/main/llm_filetuning/LLM_TUNING_20251005_FINAL.ipynb)

This implementation is optimized for **Google Colab** using a **T4 GPU**, enabling efficient training with limited resources.

---

## 🎯 Project Objectives

- Fine-tune **Mistral-7B** using **Parameter-Efficient Fine-Tuning (PEFT)**.  
- Apply **LoRA with 4-bit quantization (QLoRA)** to reduce memory footprint.  
- Train and evaluate the model on a **subset of the AG News dataset**.  
- Assess model performance using:
  - **Training & Validation Loss**
  - **Accuracy and F1 Score**

---

## ⚙️ Technical Configuration

### Model and Frameworks
- **Base Model:** Mistral-7B (from Hugging Face Transformers)
- **Fine-Tuning Frameworks:** PEFT + BitsAndBytes (for 4-bit quantization)
- **Key Libraries:**
  - `transformers`
  - `datasets`
  - `peft`
  - `bitsandbytes`
  - `accelerate`
  - `evaluate`
  - `scikit-learn`

### Dataset Information
- **Dataset:** [AG News dataset](https://huggingface.co/datasets/ag_news)
- **Categories:**
  - 0 → World
  - 1 → Sports
  - 2 → Business
  - 3 → Sci/Tech
- A small subset is used to demonstrate the fine-tuning process within Colab resource limits.

### Training Environment
- **Platform:** Google Colab
- **GPU:** NVIDIA T4 (16 GB VRAM)
- **Precision:** 4-bit quantization using BitsAndBytes
- **LoRA Parameters:**
  - Rank = 8
  - Alpha = 16
  - Dropout = 0.05
- **Epochs:** Configurable (default: 3–5)
- **Batch Size:** Automatically adjusted for available VRAM

---

## 📊 Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **Training Loss** | Measures model learning progress during training |
| **Validation Loss** | Evaluates generalization during training |
| **Accuracy** | Percentage of correct predictions on test data |
| **F1 Score** | Macro-averaged harmonic mean of precision and recall |

