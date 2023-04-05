import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import AutoTokenizer
import numpy as np
import prepare_data
import math
from pathlib import Path
import json
import codecs

class MyMultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, key_dim):
        super().__init__()
        self.embedding_dim = key_dim
        self.wq = layers.Dense(self.embedding_dim)
        self.wk = layers.Dense(self.embedding_dim)
        self.wv = layers.Dense(self.embedding_dim)
        self.ff = layers.Dense(self.embedding_dim)
        self.num_heads = num_heads
 
    def split_heads(self, e):

        return tf.reshape(e, (-1, self.num_heads, e.shape[1], int(self.embedding_dim/self.num_heads)))
    
    def call(self, x, _, use_causal_mask=False): #x dim: [batch_size, input_length, vocab_size]

        q = self.split_heads(self.wq(x)) #q, k, v dim: [batch_size, heads, input_length, dk*heads]
        k = self.split_heads(self.wk(x))
        v = self.split_heads(self.wv(x))

        w = (tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])))/math.sqrt(int(self.embedding_dim/self.num_heads)) # w dim: [batch_size, heads, input_length, input_length]
    
        # masked attention
        if use_causal_mask:
            mask = tf.experimental.numpy.triu(tf.ones((1, 1, x.shape[1], x.shape[1])), 1)*-10.0e10
            w = w + mask
        w = tf.nn.softmax(w)

        z = w @ v
    
        z = tf.reshape(z, (-1, x.shape[1], self.embedding_dim))
        z = self.ff(z)
        return z




class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MyMultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attention_output = self.att(inputs, inputs, use_causal_mask=True)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

hparams = {
    "vocab_size": 50257, # from tokenizer  
    "maxlen" : 80,  # Max sequence size
    "embed_dim" : 256,  # Embedding size for each token
    "num_heads" : 1,  # Number of attention heads
    "num_layers": 1,
    "feed_forward_dim": 256  # Hidden layer size in feed forward network inside transformer
}

batch_size = 128

vocab_size = hparams["vocab_size"]
maxlen = hparams["maxlen"]
embed_dim = hparams["embed_dim"]
num_heads = hparams["num_heads"]
num_layers = hparams["num_layers"]
feed_forward_dim = hparams["feed_forward_dim"]

model_name = f"test_h{num_heads}l{num_layers}emb{embed_dim}"
model_folder = "models/" + model_name
Path(model_folder).mkdir(parents=True, exist_ok=True)
with codecs.open (model_folder + "/params.json", "w", "utf-8") as f:
    f.write(json.dumps(hparams))

def create_model():
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_blocks = keras.Sequential([TransformerBlock(embed_dim, num_heads, feed_forward_dim) for _ in range(num_layers)])
    x = transformer_blocks(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam", loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model







class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
        self, input_length, max_tokens, start_prompt, file = None, top_k=10, print_every=1
    ):
        self.max_tokens = max_tokens
        self.input_length = input_length
        self.print_every = print_every
        self.k = top_k
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.start_tokens = self.tokenizer.encode(start_prompt)
        self.file = file

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)


    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = self.input_length - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.input_length]
                sample_index = self.input_length - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y = self.model.predict(x, verbose=0)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = self.tokenizer.decode(self.start_tokens + tokens_generated)
        print(f"\ngenerated text:\n{txt}\n")

        if self.file:
            with open(self.file, "a") as f:
                f.write(txt + "\n")



model = create_model()
text_gen_callback = TextGenerator(maxlen, 50, "And then he said ", file=model_folder + "/generated_samples.txt")


checkpoint_path = f"models/{model_name}/ckpt.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
epochs = 25

for epoch in range(1, epochs + 1):
    print("epoch: " , epoch)
    X_test, Y_test, X_train, Y_train = prepare_data.getTestTrain(maxlen, 0.05)
    model.fit(X_train[:50*batch_size], Y_train[:50*batch_size], verbose=1, epochs=1, batch_size=batch_size, callbacks=[text_gen_callback, cp_callback])