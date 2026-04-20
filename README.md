# 📦 Evian: Explainable Visual Instruction-tuning Data Auditing

Official implementation of  
**“Evian: Towards Explainable Visual Instruction-tuning Data Auditing”**

---

## 🚀 Overview

EVIAN is an **interpretable data auditing pipeline** for vision-language datasets.

Unlike traditional methods that use **coarse scores**, EVIAN performs **fine-grained, explainable evaluation** by decomposing responses and scoring them along multiple dimensions.

---

## 🧠 Method

### 🔹 Decomposition-then-Evaluation

Each response is decomposed into:

- Visual description  
- Inference (`<INFER>`)  
- Knowledge (`<KNOW>`)

Then evaluated along:

- **Logical Coherence ($S_L$)**
- **Factual Accuracy ($S_K$)**
- **Image-Text Consistency ($S\_V$)**

---

## 📊 Key Insights

- Small **high-quality data > large noisy data**
- **Logical Coherence is critical**
