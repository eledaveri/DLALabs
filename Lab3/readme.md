# Lab3: Transformers

## 📖 Theory
Transformers represent a breakthrough in natural language processing by enabling models to learn contextual relationships between words through the **self-attention mechanism**.  
Self-attention computes a weighted representation of tokens based on their relevance to each other. Each token is projected into **query (Q)**, **key (K)**, and **value (V)** vectors, and attention scores are computed as:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

This mechanism allows the model to capture both local and long-range dependencies without relying on recurrence or convolution.

**Pre-trained models**, such as **BERT** and **DistilBERT**, are trained on massive corpora using unsupervised objectives, and can later be adapted to downstream tasks like sentiment analysis.

- **DistilBERT** is a smaller and faster variant of BERT, obtained through **knowledge distillation**, where a large “teacher” model (BERT) transfers knowledge to a smaller “student” model.  
  - Reduces the number of layers from 12 to 6.  
  - Has about *40% fewer parameters* and runs *~60% faster*.  
  - Retains *over 95% of BERT’s accuracy* on benchmark tasks.  
  - Omits some components such as the next-sentence prediction head, focusing only on the masked language modeling objective.  

This laboratory explores how **transfer learning** with Transformers can be applied to binary sentiment classification.  
- First, DistilBERT is used as a frozen feature extractor with a classical classifier.  
- Next, it is fine-tuned end-to-end for task-specific adaptation.  
- Finally, **parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation)** is introduced, which updates only a small subset of parameters, reducing both training cost and memory footprint.

---

## ⚙️ Implementation
The notebook follows three main stages:

### 1. Dataset
- **Cornell Rotten Tomatoes dataset**, loaded with HuggingFace `datasets`, contains movie reviews divided in negative (0) and positive (1)
- Texts tokenized with DistilBERT’s tokenizer.  

### 2. Baseline: DistilBERT as Frozen Feature Extractor
- Extract embeddings from the `[CLS]` token using HuggingFace `pipeline`.  
- Train a **Linear SVM** classifier on top of these embeddings.  

### 3. Fine-tuning
- DistilBERT fine-tuned end-to-end using HuggingFace `Trainer`.  
- Training arguments, collators, and evaluation metrics defined (accuracy, precision, recall, F1).  

### 4. LoRA (Parameter-Efficient Fine-Tuning)
- Integrated via the HuggingFace `peft` library.  
- Updates only low-rank adaptation matrices inside Transformer layers.  
- Reduces number of trainable parameters while keeping accuracy close to full fine-tuning.  

---

## 📊 Results

### Baseline (DistilBERT + LinearSVC) 
| Dataset      | Accuracy | Precision | Recall | F1-score |
|--------------|----------|-----------|--------|----------|
| Validation   | 0.82     | 0.82      | 0.82   | 0.82     |
| Test         | 0.80     | 0.80      | 0.80   | 0.80     |

### Fine-tuned DistilBERT
| Dataset      | Accuracy | Precision | Recall | F1-score |
|--------------|----------|-----------|--------|----------|
| Validation   | 0.85     | 0.84      | 0.86   | 0.85     |
| Test         | 0.83     | 0.80      | 0.89   | 0.84     |

### Fine-tuned DistilBERT with LoRA
| Dataset      | Accuracy | Precision | Recall | F1-score |
|--------------|----------|-----------|--------|----------|
| Validation   | 0.86     | 0.85      | 0.86   | 0.86     |
| Test         | 0.83     | 0.82      | 0.85   | 0.84     |

---

## 📝 Key Takeaways
- The **baseline** with frozen DistilBERT embeddings and LinearSVC provides a reasonable starting point.  
- **Fine-tuning DistilBERT** improves all metrics, showing the benefit of adapting the model parameters to the task.  
- **LoRA** achieves performance comparable to full fine-tuning while updating far fewer parameters, making it a cost-effective alternative.  
- This workflow demonstrates the flexibility and efficiency of modern Transformer-based NLP pipelines, from feature extraction to parameter-efficient fine-tuning.  

---
