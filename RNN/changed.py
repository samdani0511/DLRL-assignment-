import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

# --- Input text ---
text = "The beautiful girl whom I met last time is very intelligent also"
# text = "The handsome boy whom I met last time is very intelligent also"

# --- Character mapping ---
chars = sorted(list(set(text)))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}
num_chars = len(chars)

# --- Sequence preparation ---
seq_length = 5
sequences = []
labels = []

for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[c] for c in seq])
    labels.append(char_to_index[label])

X = np.array(sequences)
y = np.array(labels)

# --- One-hot encoding ---
X_one_hot = to_categorical(X, num_classes=num_chars)
y_one_hot = to_categorical(y, num_classes=num_chars)

# --- Model building ---
rnn_units = 128

model = Sequential([
    SimpleRNN(rnn_units, input_shape=(seq_length, num_chars), activation='tanh'),
    Dense(num_chars, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Model training ---
epochs = 200
model.fit(X_one_hot, y_one_hot, epochs=epochs, batch_size=8, verbose=2)

# --- Function for temperature-based sampling ---
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# --- Text generation ---
def generate_text(model, seed_text, length=100, temperature=0.8):
    generated = seed_text
    seed_text = seed_text[-seq_length:]  # Ensure proper length
    
    for _ in range(length):
        x = np.array([[char_to_index.get(c, 0) for c in seed_text]])
        x_one_hot = to_categorical(x, num_classes=num_chars)
        preds = model.predict(x_one_hot, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]
        generated += next_char
        seed_text = generated[-seq_length:]
    return generated

# --- Generate text ---
seed = "The handsome boy whom I met "
generated_text = generate_text(model, seed, length=100, temperature=0.8)

print("Generated Text:")
print(generated_text)
