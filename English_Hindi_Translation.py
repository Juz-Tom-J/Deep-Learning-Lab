import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model

eng_hin = <location of your dataset>
data=pd.read_csv(eng_hin, encoding='utf-8')

# split here ------------------

# Get English and Hindi Vocabulary
all_eng_words = set()
for eng in data['English']:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

all_hin_words = set()
for hin in data['Hindi']:
    for word in hin.split():
        if word not in all_hin_words:
            all_hin_words.add(word)

data['len_eng_sen'] = data['English'].apply(lambda x: len(x.split(" ")))
data['len_hin_sen'] = data['Hindi'].apply(lambda x: len(x.split(" ")))

data = data[data['len_eng_sen'] <= 20]
data = data[data['len_hin_sen'] <= 20]

max_len_src = max(data['len_hin_sen'])
max_len_tar = max(data['len_eng_sen'])

inp_words = sorted(list(all_eng_words))
tar_words = sorted(list(all_hin_words))
num_enc_toks = len(all_eng_words)
num_dec_toks = len(all_hin_words) + 1  # for zero padding

inp_tok_idx = dict((word, i + 1) for i, word in enumerate(inp_words))
tar_tok_idx = dict((word, i + 1) for i, word in enumerate(tar_words))
rev_inp_char_idx = dict((i, word) for word, i in inp_tok_idx.items())
rev_tar_char_idx = dict((i, word) for word, i in tar_tok_idx.items())


# split here ------------------


# Split the data into train and test
X, y = data['English'], data['Hindi']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Increase batch size
batch_size = 256

# Generate batch data
def generate_batch(X=X_train, y=y_train, batch_size=batch_size):
    while True:
        for j in range(0, len(X), batch_size):
            enc_inp_data = np.zeros((batch_size, max_len_src), dtype='float32')
            dec_inp_data = np.zeros((batch_size, max_len_tar), dtype='float32')
            dec_tar_data = np.zeros((batch_size, max_len_tar, num_dec_toks), dtype='float32')
            for i, (inp_text, tar_text) in enumerate(zip(X[j:j + batch_size], y[j:j + batch_size])):
                for t, word in enumerate(inp_text.split()):
                    enc_inp_data[i, t] = inp_tok_idx[word]
                for t, word in enumerate(tar_text.split()):
                    if t < len(tar_text.split()) - 1:
                        dec_inp_data[i, t] = tar_tok_idx[word]
                    if t > 0:
                        dec_tar_data[i, t - 1, tar_tok_idx[word]] = 1.0
            yield [enc_inp_data, dec_inp_data], dec_tar_data


# split here ------------------


# Encoder-Decoder Architecture
latent_dim = 250

# Encoder
enc_inps = Input(shape=(None,))
enc_emb = Embedding(num_enc_toks, latent_dim, mask_zero=True)(enc_inps)
enc_lstm = LSTM(latent_dim, return_state=True)
enc_outputs, st_h, st_c = enc_lstm(enc_emb)
enc_states = [st_h, st_c]

# Set up the decoder
dec_inps = Input(shape=(None,))
dec_emb_layer = Embedding(num_dec_toks, latent_dim, mask_zero=True)
dec_emb = dec_emb_layer(dec_inps)
dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
dec_outputs, _, _ = dec_lstm(dec_emb, initial_state=enc_states)
dec_dense = Dense(num_dec_toks, activation='softmax')
dec_outputs = dec_dense(dec_outputs)


# split here ------------------


# Define the model
model = Model([enc_inps, dec_inps], dec_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')  # Use Adam optimizer for faster convergence

# split here ------------------


train_samples = len(X_train)
val_samples = len(X_test)

# Train the model with a larger batch size
model.fit(x=generate_batch(X_train, y_train, batch_size=batch_size),
          steps_per_epoch=train_samples // batch_size,
          epochs=50,
          validation_data=generate_batch(X_test, y_test, batch_size=batch_size),
          validation_steps=val_samples // batch_size)


# split here ------------------


# Encode the input sequence to get the "thought vectors"
enc_model = Model(enc_inps, enc_states)

# Decoder setup
# Below tensors will hold the states of the previous time step
dec_st_inp_h = Input(shape=(latent_dim,))
dec_st_inp_c = Input(shape=(latent_dim,))
dec_states_inps = [dec_st_inp_h, dec_st_inp_c]

dec_emb2= dec_emb_layer(dec_inps) # Get the embeddings of the decoder sequence

# To predict the next word in the sequence, set the initial states to the states from the previous time step
dec_outputs2, st_h2, st_c2 = dec_lstm(dec_emb2, initial_state=dec_states_inps)
dec_states2 = [st_h2, st_c2]
dec_outputs2 = dec_dense(dec_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

# Final decoder model
dec_model = Model(
    [dec_inps] + dec_states_inps,
    [dec_outputs2] + dec_states2)


# split here ------------------


def translate(inp_seq):
    # Encode the input as state vectors.
    states_value = enc_model.predict(inp_seq)
    # Generate empty target sequence of length 1.
    tar_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    tar_seq[0, 0] = tar_tok_idx['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_cond = False
    dec_sen = ''
    while not stop_cond:
        output_toks, h, c = dec_model.predict([tar_seq] + states_value)

        # Sample a token
        sampled_tok_idx = np.argmax(output_toks[0, -1, :])
        sampled_char = rev_tar_char_idx[sampled_tok_idx]
        dec_sen += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(dec_sen) > 50):
            stop_cond = True

        # Update the target sequence (of length 1).
        tar_seq = np.zeros((1,1))
        tar_seq[0, 0] = sampled_tok_idx

        # Update states
        states_value = [h, c]

    return dec_sen

train_gen = generate_batch(X_train, y_train, batch_size = 1)
k=0
(inp_seq, actual_output), _ = next(train_gen)
hin_sen = translate(inp_seq)
print(f'''Input English sentence: {X_train[k:k+1].values[0]}\n
          Predicted Hindi Translation: {hin_sen[:-4]}\n
          Actual Hindi Translation: {y_train[k:k+1].values[0][6:-4]}''')
