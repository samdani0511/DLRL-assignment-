# Character-Level RNN Text Generation

## Overview
This project implements a character-level Recurrent Neural Network (RNN) to generate text based on a seed sentence. The model learns character patterns from a small input text and predicts the next character iteratively, producing coherent sequences.  

The code has been improved from the original version to enhance performance, readability, and text generation diversity.

---

## Features

- Character-level RNN for sequence prediction.
- Temperature-controlled sampling for diverse text generation.
- Supports flexible seed text and output length.
- Handles small datasets efficiently.
- Clean, modular code structure with reusable functions.

---

## Improvements in This Version

1. **RNN Architecture**
   - Increased RNN units from 50 → 128 to capture more complex patterns.
   - `tanh` activation used, standard for RNNs.

2. **Efficient One-Hot Encoding**
   - Replaced manual `tf.one_hot` with `keras.utils.to_categorical`.

3. **Temperature-Based Sampling**
   - Generates more natural and diverse text.
   - Allows tuning randomness in predictions.

4. **Text Generation Function**
   - `generate_text()` supports custom seed text, length, and temperature.

5. **Training Enhancements**
   - Epochs increased from 100 → 200 for better convergence.
   - Batch size reduced to 8 for small datasets.

6. **Seed Handling**
   - Automatically manages seeds shorter than sequence length.
   - Sliding window approach ensures proper input preparation.

7. **Improved Code Structure**
   - Modular functions for sampling and generation.
   - Better readability and maintainability.

---

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib (optional, for visualization)

Install dependencies via pip:

```bash
pip install tensorflow keras numpy matplotlib



<img width="628" height="183" alt="image" src="https://github.com/user-attachments/assets/cb7bf50b-f0a9-4b02-aad7-a5a4e8d0f5fd" />


Generated Text:
The handsome boy whom I met last time is very intelligent alsotitlmesit nmst s me yl vtel igenl alio it elso enmellsaenntulligln
