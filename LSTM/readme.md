# Airline Passengers LSTM Prediction – Model Updates

## Model Changes & Improvements

1. **Dataset Integration via KaggleHub**
   - Replaced local CSV file with automatic download using `kagglehub`.
   - Handles file paths dynamically for Colab compatibility.

2. **LSTM Architecture**
   - Increased LSTM neurons from **10 → 50** for better learning capacity.
   - Removed `return_sequences=True` as only one LSTM layer is used.
   - Output layer remains a single neuron (`Dense(1)`) for regression.

3. **Training Enhancements**
   - Increased **epochs from 50 → 100** for improved convergence.
   - Added **validation split (0.1)** during training for better monitoring.
   - Batch size kept at 1 for fine-grained weight updates.

4. **Sequence Preparation**
   - Created a helper function `create_sequences()` to reduce code repetition.
   - Maintains a **time step of 10** for input sequences.

5. **Colab Compatibility**
   - All file paths and plotting compatible with Google Colab.
   - Model summary and plot saved automatically (`model_plot.png`).

6. **Evaluation**
   - RMSE calculation remains the same but applied to scaled inverse predictions.
   - Predictions plotted alongside actual values for visual comparison.

<img width="850" height="393" alt="image" src="https://github.com/user-attachments/assets/6abac769-251b-4d76-8759-ae6fd673198b" />


<img width="654" height="197" alt="image" src="https://github.com/user-attachments/assets/b35168c6-3b14-4a80-84d4-425e3c518845" />
