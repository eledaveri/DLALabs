# Lab3_Transformers

## 🔹 Theory
Transformers represent a breakthrough in natural language processing by enabling models to learn contextual relationships between words through the self-attention mechanism. Pre-trained models, such as BERT and DistilBERT, are trained on massive corpora using unsupervised objectives, and can later be adapted to downstream tasks like sentiment analysis.
*DistilBERT* is a smaller and faster variant of BERT, obtained through knowledge distillation, a technique where a large “teacher” model (BERT) transfers knowledge to a smaller “student” model. Specifically, DistilBERT reduces the number of layers from 12 to 6 while preserving most of BERT’s performance. Despite having about *40% fewer parameters* and running approximately *60% faster*, it retains *over 95% of BERT’s accuracy* on benchmark tasks. Architecturally, DistilBERT maintains the Transformer encoder structure (multi-head self-attention, feed-forward layers, residual connections, and layer normalization), but omits some components such as the next-sentence prediction head used in BERT’s pre-training. Instead, it focuses solely on the masked language modeling objective, which proves sufficient to learn strong contextual representations.
This laboratory exercise explores how transfer learning with Transformers can be applied to a binary sentiment classification problem. Initially, the model serves as a frozen feature extractor combined with a classical classifier. Subsequently, the same Transformer is fine-tuned end-to-end, allowing task-specific adaptation and performance improvements. To further optimize the fine-tuning process, **parameter-efficient methods such as LoRA (Low-Rank Adaptation)** are introduced, which allow adapting large models by updating only a small fraction of their parameters, reducing both training cost and memory footprint.   

## 🔹 Implementation
The notebook first loads the **Cornell Rotten Tomatoes dataset** using the HuggingFace `datasets` library. DistilBERT is initialized with its corresponding tokenizer, and features are extracted from the `[CLS]` token embeddings using the HuggingFace `pipeline`. These representations are used to train a **Linear Support Vector Machine (SVM)** baseline model. In the second phase, the dataset is tokenized and prepared for direct input into DistilBERT. Using the HuggingFace `Trainer` API, DistilBERT is fine-tuned on the sentiment classification task with labeled data. Training arguments, data collators, and evaluation metrics (accuracy, precision, recall, F1) are defined to monitor progress and ensure reproducibility. Additionally, the notebook integrates **LoRA via the `peft` library**, which modifies only selected projection matrices of the Transformer layers. This enables efficient adaptation with fewer trainable parameters, making fine-tuning feasible even on limited hardware resources.

## 🔹 Results

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

The baseline experiment with frozen DistilBERT embeddings and a LinearSVC classifier yields reasonable performance on both validation and test sets, establishing a starting point. Fine-tuning DistilBERT on the dataset demonstrates improved results, with higher accuracy and better-balanced classification metrics. This outcome highlights the advantage of allowing Transformer parameters to adapt to the specific sentiment classification task.
The application of **LoRA** shows that competitive performance can be reached while drastically reducing the number of parameters to train, highlighting the practical advantages of parameter-efficient fine-tuning. Overall, the notebook illustrates a complete workflow: from data loading and exploration, through baseline feature extraction, to full fine-tuning with HuggingFace and LoRA, showing both the flexibility and the efficiency of modern Transformer-based NLP pipelines.
Overall, the notebook illustrates a complete workflow: from data loading and exploration, through baseline feature extraction, to full fine-tuning with HuggingFace tools, showing both the flexibility and the power of modern Transformer-based NLP pipelines.
