1. WHAT WAS CHANGED?

Original AlexNet → Improved AlexNet

- Added Batch Normalization after major convolution layers
- Changed first layer padding from 'valid' to 'same'
- Used He Normal initialization for better ReLU convergence
- Replaced Flatten + 2×4096 Dense layers with Global Average Pooling
- Reduced classifier size (lighter & faster)

2. WHY THESE CHANGES ARE BETTER

(1) Batch Normalization
- Stabilizes training
- Faster convergence
- Allows higher learning rates

(2) 'same' Padding in First Layer
- Preserves spatial information
- Prevents aggressive feature loss at early stage

(3) He Normal Initialization
- Designed for ReLU networks
- Prevents vanishing/exploding gradients

(4) Global Average Pooling (GAP)
- Reduces parameters drastically
- Less overfitting
- Better generalization

Original AlexNet Parameters ≈ 60 million
Improved Version ≈ ~15 million (approx.)

(5) Smaller Fully Connected Layer
- Faster training
- Lower memory usage
- Suitable for edge devices (Raspberry Pi, Jetson, Mobile)

3. PERFORMANCE IMPACT

| Aspect              | Original AlexNet | Improved AlexNet |
|---------------------|------------------|------------------|
| Parameters          | Very High        | Much Lower       |
| Training Stability  | Medium           | High             |
| Overfitting Risk    | High             | Reduced          |
| Deployment Ready    | No               | Yes              |

4. WHEN TO USE THIS VERSION

- Academic projects
- Embedded / Edge AI
- Faster experimentation
- Limited GPU / RAM systems


<img width="654" height="587" alt="image" src="https://github.com/user-attachments/assets/f7fdd5a3-18b7-4a5b-b5d7-749ae4c7011e" />

