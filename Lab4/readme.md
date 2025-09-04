# Lab 4: OOD Detection & Adversarial Learning

## 📖 Theory

The lab focuses on two main aspects:

### 1. Out-of-Distribution (OOD) Detection
- A model trained on a fixed dataset (e.g., CIFAR-10) tends to output **overconfident predictions** even for samples not belonging to the training distribution.  
- **OOD detection** aims to detect such inputs and reject them rather than misclassify.  
- These are the main approaches explored:
  - **Softmax confidence**: using the maximum predicted probability as a score.  
  - **Autoencoders**: reconstructing inputs. Since they only learn to reconstruct ID samples, OOD images yield higher reconstruction errors.  
  - **ODIN**: a technique that improves OOD detection by applying **temperature scaling** to softmax outputs and adding a **small perturbation** to increase the gap between ID and OOD scores.

2. **Adversarial Learning**
   - Deep learning models are vulnerable to **adversarial attacks**, where small imperceptible perturbations cause misclassifications.
   - One of the simplest methods to generate adversarial examples is **FGSM (Fast Gradient Sign Method)**, which adds a perturbation to the original input proportional to the gradient of the loss with respect to the input.
   - **Adversarial training** improves robustness by injecting adversarial examples during training, making the model less sensitive to such perturbations (at the cost of a small drop in clean accuracy)

---

## ⚙️ Implementation

1. **Dataset**
   - **ID (in-distribution):** CIFAR-10
   - **OOD:** synthetic datasets generated with `torchvision.datasets.FakeData`
   - **SVHN** dataset used as Out Of Distribution

2. **Preprocessing**
   - Image normalization
   - DataLoader for train/test sets and OOD datasets

3. **Models**
   - **Simple CNN** a shallow CNN used for image classification
   - **Deep CNN** a more complex implementation of a CNN for image classification
   - **ResNet** pretrained on CIFAR
   - **Autoencoder** with convolutional layers to compress and reconstruct images

4. **OOD pipeline**
   - Train the CNN on CIFAR-10
   - Collect predictions and confidence scores
   - Compare score distributions for ID vs OOD using histograms
   - Compute metrics: **AUROC, False Poaitive Rate - True Positive Rate and Precision-Recall cuve**

5. **OOD pipeline with Autoencoder**
   - Train the autoencoder on CIFAR-10
   - Compute reconstruction error on ID and OOD samples
   - Use the error as OOD score → threshold to classify a sample as ID/OOD

6. **Adversarial Training (FGSM)**
   - Generate adversarial images using **FGSM**:
     ```
     x_adv = x + epsilon * sign(gradient_x(L(theta, x, y)))
     ```
   - Train the model with both original and perturbed images
   - Compare performance with and without adversarial training
  
7. **ODIN Out-of-Distribution detector for Neural networks**
   - Improves detection of OOD samples using a pre-trained classifier.
   - Applies temperature scaling to softmax outputs to sharpen confidence differences.
   - Adds a small input perturbation to increase predicted confidence for in-distribution data.
   - Helps separate ID and OOD samples without retraining the model.

9. **Reproducibility**
   - Fixed seeds for PyTorch, NumPy, and Python
   - Use of deterministic algorithms

---

## 📊 Results

- **Base CNN:** trained on CIFAR-10, achieving good accuracy on the validation set.
- **OOD detection with softmax scores:**
   - **Softmax scores (CNN):** partial separation of ID/OOD distributions, limited sensitivity  
   - **Autoencoder:** clear separation, reconstruction error significantly higher for OOD samples  
   - **ODIN:**  
   - Best performance with **T=100, ε=0.004**  
   - Achieved **AUROC ≈ 0.89**, a significant improvement over softmax baseline  

- **Autoencoder:** provided a clearer separation between ID and OOD due to high reconstruction errors on OOD images.
- **FGSM:** the original model was vulnerable to the generated perturbations.
- **Adversarial training:** increased CNN robustness against FGSM, reducing accuracy drop under attack, at the cost of a slight decrease on clean images.
  - **Without adversarial training (Deep CNN):**
     - ε=0.00000 → Accuracy **99.1%**  
      - ε=0.00392 → Accuracy **35.6%**  
      - ε=0.01961 → Accuracy **29.0%**  
  → Model is highly vulnerable, accuracy collapses with small perturbations.  

  - **With adversarial training:**  
      - Clean accuracy slightly reduced (**~89.2%**)  
      - Robustness improved against FGSM:
       - ε=0.00392 → Accuracy **35.3%**  
       - ε=0.01176 → Accuracy **27.4%**  
  → Still vulnerable, but degradation is slower and more stable compared to the non-robust model.  


## 📝 Key Takeaways
- **Softmax-based OOD detection** is weak, while **autoencoders** and **ODIN** provide more reliable detection.  
- **FGSM attacks** expose strong vulnerabilities of CNNs; even tiny perturbations reduce accuracy drastically.  
- **Adversarial training** trades a slight drop in clean performance for significantly improved robustness.  
- Combining **OOD detection methods** with **robust training** is essential for real-world deployment of deep learning models in safety-critical scenarios.  

