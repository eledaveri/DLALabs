# Lab 4: OOD Detection & Adversarial Learning

## 🔹 Theory

The lab focuses on two main aspects:

1. **Out-of-Distribution (OOD) Detection**
   - A classifier can be highly accurate on training data but unreliable on “out-of-distribution” samples (e.g., images of classes not present in CIFAR-10).
   - The goal is to equip the system with a mechanism capable of recognizing and rejecting OOD samples.

2. **Adversarial Learning**
   - Deep learning models are vulnerable to **adversarial attacks**, where small imperceptible perturbations cause misclassifications.
   - One of the simplest methods to generate adversarial examples is **FGSM (Fast Gradient Sign Method)**, which adds a perturbation to the original input proportional to the gradient of the loss with respect to the input.
   - Including adversarial examples in training (*adversarial training*) improves model robustness.

3. **Autoencoder for OOD detection**
   - **Autoencoders** learn to reconstruct images similar to those in the training set.
   - For OOD images, reconstruction is worse → the reconstruction error can be used as a score for OOD detection.

---

## 🔹 Implementation

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
   - **Autoencoder** with convolutional layers to compress and reconstruct images

4. **OOD pipeline with CNN**
   - Train the CNN on CIFAR-10
   - Collect predictions and confidence scores
   - Compare score distributions for ID vs OOD using histograms
   - Compute metrics: **AUROC, False Poaitive Rate - True Positive Rate**

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

## 🔹 Results

- **Base CNN:** trained on CIFAR-10, achieving good accuracy on the validation set.
- **OOD detection with softmax scores:** OOD samples (FakeData) show different distributions → positive AUROC and False-Positive Rate/True-positive Rate, but limited sensitivity.
- **Autoencoder:** provided a clearer separation between ID and OOD due to high reconstruction errors on OOD images.
- **FGSM:** the original model was vulnerable to the generated perturbations.
- **Adversarial training:** increased CNN robustness against FGSM, reducing accuracy drop under attack, at the cost of a slight decrease on clean images.

