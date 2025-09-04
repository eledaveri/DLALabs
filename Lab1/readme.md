# Lab1: MLPs, CNNs, and Residual Connections

## 📖 Theory

### Multilayer Perceptron (MLP)
A **Multilayer Perceptron (MLP)** is a feedforward neural network composed of fully connected layers. Input vectors are flattened and passed through a sequence of linear transformations, each followed by a non-linear activation function such as ReLU. While MLPs can model complex relationships in structured data, they struggle with image data because flattening discards spatial information. Increasing the depth (number of layers) enhances the representational capacity of the model, but very deep MLPs often face training instabilities such as vanishing gradients and optimization difficulties, which limit their effectiveness without architectural improvements like residual connections.
 

### Convolutional Neural Network (CNN)
A **Convolutional Neural Network (CNN)** is a neural architecture designed to preserve and exploit the spatial structure of data. Instead of flattening inputs as in MLPs, CNNs apply local filters (kernels) that slide across the image to capture spatial features such as edges, textures, and patterns. Pooling layers further reduce dimensionality, improving computational efficiency and providing a degree of translation invariance. By leveraging the 2D structure of images, CNNs achieve far superior performance compared to MLPs on vision tasks.


**Residual connections** allow neural networks to learn residual mappings instead of direct transformations. A residual block computes \( y = F(x) + x \), where \( F(x) \) is the learned transformation and \( x \) is the identity shortcut. This simple idea addresses vanishing gradient issues, enabling the training of much deeper models while keeping optimization stable and improving generalization performance.


### MLP with Residual Connections
An **MLP with residual connections** extends the standard multilayer perceptron by introducing skip connections between fully connected layers. These shortcuts preserve information across layers, allowing deeper MLPs to be trained efficiently without suffering from vanishing gradients. As a result, residual MLPs converge faster, remain stable at greater depths, and typically achieve higher accuracy compared to their vanilla counterparts.


### CNN with Residual Connections
A **CNN with residual connections** integrates skip connections within its convolutional blocks, following the ResNet-like design. These shortcuts enable the training of very deep convolutional networks, sometimes with dozens or even hundreds of layers, without suffering from vanishing gradients. By stabilizing optimization and improving information flow, residual CNNs consistently outperform standard CNNs, especially as network depth increases.
  

---

## ⚙️ Implementation (Notebook Overview)

The notebook `Lab1-CNNs.ipynb` contains the following steps:

### 1. Dataset & Preprocessing

This project uses two standard datasets:

- **MNIST**: used for MLP experiments.  
- **CIFAR-10**: used for CNN experiments.  

**Preprocessing steps:**
- Split training sets into **training** and **validation** subsets (5,000 samples for validation).  
- Convert images to tensors.  
- Normalize using dataset-specific mean and standard deviation:  
  - MNIST: mean = 0.1307, std = 0.3081  
  - CIFAR-10: mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)  
- Apply data augmentation for CIFAR-10 training set: random cropping with padding and horizontal flips.    


### 2. Training Pipeline

The training pipeline is implemented in **PyTorch** and includes support for early stopping, learning rate scheduling, and experiment logging (via **Weights & Biases**).

**Core components:**

- **Loss function:** Cross-Entropy Loss for classification.  
- **Optimizers:**  
  - **MLPs:** Adam  
  - **CNNs / Residual CNNs:** SGD with momentum and weight decay  
- **Scheduler:** Optional (e.g., CosineAnnealingLR) for CNNs.  
- **Early stopping:** Monitors validation loss with configurable patience and minimum improvement.  
- **Evaluation:** Computes accuracy and generates a full classification report.  
- **Logging:** Tracks per-epoch metrics such as training loss, validation loss, validation accuracy, learning rate, and best model performance. Metrics and artifacts are logged to **W&B**, including gradients, confusion matrices, and trained model weights.

**Helper functions:**

- `train()`: Performs full training loop with optional early stopping and scheduler support.  
- `evaluate()`: Evaluates a model on a dataset and optionally computes loss, accuracy, classification report, and logs to W&B.  
- `plot_training_metrics()`: Visualizes training loss, validation accuracy, and optionally validation loss over epochs.

**Workflow overview:**

1. Initialize the model (MLP, Residual MLP, CNN, or Residual CNN) with configurable depth and width.  
2. Set up the optimizer, loss function, and scheduler (if applicable).  
3. Train the model with `train()`, logging metrics per epoch.  
4. Plot metrics using `plot_training_metrics()`.  
5. Evaluate on the test set with `evaluate()`.  
6. Save model checkpoints and log artifacts to W&B.  


### 3. Implemented Models

#### Configurable MLP
- Fully connected network with **configurable depth** and hidden layer sizes.  
- Each hidden layer uses: `Linear → BatchNorm → ReLU → Dropout` for stable training and regularization.  
- Final layer maps to the number of classes (no activation, compatible with `CrossEntropyLoss`).  
- Suitable for shallow to moderately deep architectures.

#### Residual MLP
- Extension of MLP with **residual skip connections** between hidden layers.  
- Projection layers are used when input/output dimensions differ, ensuring valid residuals.  
- Improves gradient flow and training stability, allowing **much deeper networks**.  
- Final output produced by a linear layer (no activation).

#### Configurable CNN
- Built from configurable convolutional blocks: `Conv → BatchNorm → ReLU → Dropout`.  
- Pooling layers progressively reduce spatial dimensions.  
- Ends with **Global Average Pooling (GAP)** instead of large fully connected layers, reducing parameters and enabling **Class Activation Maps (CAM)**.  
- Dropout is applied before the final classifier for regularization.  
- Supports multiple depth configurations (shallow to very deep) via `hidden_channels`.

#### Residual CNN
- Uses **Residual Blocks** with skip connections, inspired by ResNet.  
- Each block contains two convolutions and an identity (or projection) shortcut.  
- Downsampling applied selectively to prevent overly small feature maps.  
- Final features are processed with GAP and a linear classifier.  
- Enables **deep architectures** (light to extremely deep) while mitigating vanishing gradients.  
- Supports **CAM visualization** through the GAP layer.
 
  


---

## 📊 Results

Below is a structured summary of the results.  

### MLP vs Residual MLP (MNIST)

| Model              | Depth | Test Accuracy | 
|--------------------|-------|---------------|
| MLP                | 1     | 95%           |
| MLP                | 3     | 94%           | 
| MLP                | 5     | 94%           | 
| MLP                | 8     | 94%           |   
| MLP                | 16    | 62%           | 
| Residual MLP       | 1     | 95%           | 
| Residual MLP       | 3     | 95%           |
| Residual MLP       | 5     | 95%           | 
| Residual MLP       | 8     | 95%           | 
| Residual MLP       | 16    | 96%           | 
| Residual MLP       | 24    | 96%           | 
| Residual MLP       | 32    | 96%           | 
| Residual MLP       | 50    | 95%           | 


#### Key observations
1. **Vanilla MLP**
   - Performs well with shallow architectures (1–8 layers, ~94–95% test accuracy).  
   - When made deeper (16 layers), performance **collapses to 62%**, caused by training instabilities such as vanishing/exploding gradients and difficulty propagating information.  
   - In short, deeper vanilla MLPs do **not** bring improvements and are actually harder to optimize.  

2. **Residual MLP**
   - Matches shallow MLP performance with fewer layers (~95%).  
   - Unlike vanilla MLP, residual connections stabilize training and allow **much deeper networks (up to 32+ layers)** without accuracy degradation.  
   - In fact, moderate depth (16–32 layers) yields a slight improvement (~96%).  
   - At extreme depth (50 layers) accuracy drops slightly (~95%), likely due to MNIST being a relatively simple dataset (overfitting or diminishing returns).  

#### Why this happens
- **Vanilla MLP**: Each layer must fully transform the input, making gradients harder to propagate as depth grows. Training becomes unstable.  
- **Residual MLP**: Each layer only learns a **residual correction** to the previous representation. Identity shortcuts improve gradient flow, prevent vanishing, and make deep networks trainable.  

#### Takeaway
- For **shallow models**, both MLP and Residual MLP perform similarly.  
- For **deeper models**, only the Residual MLP remains stable and can even outperform shallower versions.  
- Although MNIST itself does not benefit much from extreme depth, this experiment clearly shows how **residual connections are a key tool to stabilize optimization in deep neural networks**.
---

### CNN vs Residual CNN (CIFAR-10)


| Model               | Depth          | Hidden Channels                   | Test Accuracy (%) |                                   
|---------------------|----------------|-----------------------------------|-------------------|
| CNN                 | Shallow        | [32, 64]                          | 80                |
| CNN                 | Medium         | [32, 64, 128]                     | 89                | 
| CNN                 | Deep           | [32, 64, 128, 256]                | 91                | 
| CNN                 | Very Deep      | [32, 64, 128, 256, 512]           | 91                | 
| CNN                 | Ultra Deep     | [32, 64, 128, 256, 512, 512]      | 91                | 
| Residual CNN        | Light          | [64, 128, 256]                    | 91                | 
| Residual CNN        | Shallow        | [64, 64, 128, 128, 256]           | 92                | 
| Residual CNN        | Medium         | [64, 64, 128, 128, 256, 256]      | 93                | 
| Residual CNN        | Deep           | [64, 64, 128, 128, 256, 256, 512] | 93                | 
| Residual CNN        | Very Deep      | [64, 64, 128, 128, 256, 256, 512, 512] | 90–92        | 
| Residual CNN        | Extremely Deep | [64, 64, 128, 128, 256, 256, 512, 512, 512] | 91–93   | 
---

## 📝 Key Takeaways
- Increasing **depth** does not always improve performance; too deep networks may overfit or be harder to train.

- **Residual connections** can help, but only shallow or moderately deep networks benefit.

- In this experiment, vanilla deep CNNs achieved the highest test accuracy, outperforming very deep residual networks.

- Results partially align with ResNet (He et al., 2015), highlighting that residuals help training deep networks, but extreme depth can be detrimental on small datasets like CIFAR-10.

---

