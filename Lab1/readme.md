# Deep Learning Laboratory 1: MLPs, CNNs, and Residual Connections

## 📖 Theory

### Multilayer Perceptron (MLP)
A **Multilayer Perceptron** is a feedforward neural network composed of fully connected layers.  
- Input vectors are flattened and passed through a sequence of linear layers.  
- Each layer applies a linear transformation followed by a non-linear activation function (commonly ReLU).  
- MLPs are powerful for simple structured data but struggle with images due to the loss of spatial information when flattening inputs.  
- Depth (number of layers) increases representational capacity, but very deep MLPs suffer from **vanishing gradients** and optimization difficulties.  

### Convolutional Neural Network (CNN)
A **Convolutional Neural Network** introduces convolutional layers that preserve spatial structure.  
- Instead of flattening, CNNs use local filters (kernels) that slide across the image to capture spatial features such as edges, textures, and patterns.  
- Pooling layers reduce dimensionality, improving computational efficiency and invariance.  
- CNNs are far more effective than MLPs for image tasks because they exploit the 2D structure of data.  

### Residual Connections
Residual connections, introduced in [ResNet (He et al., 2015)](https://arxiv.org/abs/1512.03385), allow models to learn **residual mappings** rather than direct transformations.  
- A residual block computes:  
  \[
  y = F(x) + x
  \]
  where \(F(x)\) is the learned transformation and \(x\) is the input (identity shortcut).  
- This alleviates vanishing gradient problems and allows training of **much deeper models**.  
- Residual connections stabilize optimization and improve generalization.  

### MLP with Residual Connections
- A residual MLP introduces skip connections between fully connected layers.  
- This allows deeper MLPs to learn efficiently by preserving information across layers.  
- Compared to vanilla MLPs, residual MLPs converge faster and achieve higher accuracy.  

### CNN with Residual Connections
- A residual CNN integrates skip connections within convolutional blocks.  
- This architecture (ResNet-like) enables the training of very deep CNNs (dozens or even hundreds of layers).  
- Residual CNNs outperform standard CNNs, particularly as depth increases.  

---

## ⚙️ Implementation (Notebook Overview)

The notebook `Lab1-CNNs.ipynb` contains the following steps:

### 1. Dataset & Preprocessing
- **MNIST**: for MLP experiments.  
- **CIFAR-10**: for CNN experiments.  
- Standard transformations:
  - Convert to tensors.
  - Normalize dataset-specific mean and standard deviation.
  - Split training into train/validation subsets.

### 2. Training Pipeline
- Implemented using **PyTorch**.  
- Core components:
  - **Loss**: Cross-Entropy.  
  - **Optimizer**: Adam for MLP and SGD for CNN.  
  - **Evaluation**: Accuracy and classification report.  
  - **Logging**: Training loss and validation accuracy per epoch.  
- Helper functions for training, evaluation, and plotting validation curves.

### 3. Implemented Models
- **MLP**  
  - Simple baseline MLPs with varying depth.  
  - Custom implementation for flexibility.  

- **Residual MLP**  
  - MLP with skip connections between layers.  
  - Deeper models trained successfully compared to vanilla MLPs.  

- **CNN**  
  - Shallow, medium, deep, very deep, and ultra-deep variants.  
  - Implemented with convolutional, pooling, and fully connected layers.  

- **Residual CNN**  
  - Light to extremely deep variants with ResNet-like blocks.  


---

## 📊 Results

Below is a structured summary of the results.  

### MLP vs Residual MLP (MNIST)

| Model              | Depth | Test Accuracy | Notes                          |
|--------------------|-------|---------------|--------------------------------|
| MLP (baseline)     | 3     | 94%           | Simple shallow MLP.            |
| MLP (depth: 16)    | 5     | 62%           | Performance degrades with depth. |
| Residual MLP       | 5     | 95%           | Residuals stabilize training.  |
| Residual MLP       | 16    | 96%           | Better than deep vanilla MLP.  |

---

### CNN vs Residual CNN (CIFAR-10)

| Model                  | Depth/Variant   | Test Accuracy | Notes                                 |
|-------------------------|-----------------|---------------|---------------------------------------|
| CNN (shallow)           | Small           | 79%           | Base CNN, acceptable performance.                       |
| CNN (deep)              | Large           | 83%           | Deeper improves performance         |
| Residual CNN (shallow)  | Small           | 80%           | Outperforms equivalent CNN.           |
| Residual CNN (deep)     | Large           | 77%           | Deeper worsens, possible overfitting    |
| Residual CNN (extreme)  | Very deep       | 75%           | Too deep for CIFAR-10 |

---

## 📝 Key Takeaways
- Increasing **depth** does not always improve performance; too deep networks may overfit or be harder to train.

- **Residual connections** can help, but only shallow or moderately deep networks benefit.

- In this experiment, vanilla deep CNNs achieved the highest test accuracy, outperforming very deep residual networks.

- Results partially align with ResNet (He et al., 2015), highlighting that residuals help training deep networks, but extreme depth can be detrimental on small datasets like CIFAR-10.

---

