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
import argparse


class AttentionHead(layers.Layer):
    def __init__(self, embed_dim, head_embed_dim, rate=0.5):
        super().__init__()
        self.head_embed_dim = head_embed_dim
        self.wq = layers.Dense(head_embed_dim)
        self.wk = layers.Dense(head_embed_dim)
        self.wv = layers.Dense(head_embed_dim)

    def call(self, x, use_causal_mask=True, values_only=False):
        q = self.wq(x) #q, k, v dim: [batch_size, input_length, head_embedding_dim]
        k = self.wk(x)
        v = self.wv(x)

        w = (tf.matmul(q, tf.transpose(k, perm=[0, 2, 1])))/math.sqrt(int(self.head_embed_dim)) # w dim: [batch_size, input_length, input_length]
        if use_causal_mask:
            mask = tf.experimental.numpy.triu(tf.ones((1, x.shape[1], x.shape[1])), 1)*-10.0e10
            w = w + mask
        w = tf.nn.softmax(w)


        if values_only:
            w = tf.eye(x.shape[1], batch_shape=[x.shape[0]])
            #w = tf.zeros(w.shape)
        self.w = w
        z = tf.matmul(w, v)
        return z #z: [batch_size, input_length, head_embedding_dim]

class AttentionLayer(layers.Layer):
    def __init__(self, num_heads, embedding_dim, rate=0.5):
        super().__init__()

        self.head_embedding_dim = embedding_dim//num_heads
        self.heads = [AttentionHead(embedding_dim, self.head_embedding_dim) for _ in range(num_heads)]
        self.ff = layers.Dense(embedding_dim, activation="gelu")
        self.dropout = layers.Dropout(rate)

    def call(self, x, use_causal_mask=True, use_head = None, values_only=False): #x dim: [batch_size, input_length, embed_size]


        outputs = tf.zeros((x.shape[0], x.shape[1], self.head_embedding_dim))
        if use_head is not None:
            outputs += self.heads[use_head](x)
        for head in self.heads:
            outputs += head(x, values_only=values_only)

        outputs = self.ff(outputs)
        outputs = self.dropout(outputs)

        return outputs


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.5):
        super().__init__()
        self.att = AttentionLayer(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, scale=True, center=True)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, scale=True, center=True)


    def call(self, x, use_head = None, values_only=False):
   
        a = self.att(self.layernorm1(x), use_causal_mask=True, use_head=use_head, values_only=values_only)
        x = x + a

        m = self.ffn(self.layernorm2(x))
        x = x + m
        return x



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

class TransformerModel(tf.keras.Model):

    def __init__(self, hparams):
        super().__init__()
        vocab_size = hparams["vocab_size"]
        maxlen = hparams["maxlen"]
        embed_dim = hparams["embed_dim"]
        num_heads = hparams["num_heads"]
        num_layers = hparams["num_layers"]
        feed_forward_dim = hparams["feed_forward_dim"]
        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.blocks = [TransformerBlock(embed_dim, num_heads, feed_forward_dim) for _ in range(num_layers)]
        self.out = layers.Dense(vocab_size)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6, scale=True, center=True)

    def call(self, x, use_layer = None, use_head = None, values_only=False):
        x = self.embedding(x)

        if use_layer is None:
            max_layer = len(self.blocks)
        else:
            max_layer = use_layer + 1

        for i in range(max_layer):
            x = self.blocks[i](x, use_head=use_head, values_only=values_only)

        x = self.layernorm(x)
        x = self.out(x)
        return x





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


    def generate(self):
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
            y = self.model(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        return self.tokenizer.decode(self.start_tokens + tokens_generated)
    def on_epoch_end(self, epoch, logs=None):
        txt = self.generate()
        print(f"\ngenerated text:\n{txt}\n")

        if self.file:
            with codecs.open(self.file, "a", "utf-8") as f:
                f.write(txt + "\n")



if __name__ == "__main__":

    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("--start_over", help="Clear model and start over", action='store_const', default=False, const=True)
    args = parser.parse_args()


    hparams = {
        "vocab_size": 50257, # from tokenizer  
        "maxlen" : 32,  # Max sequence size
        "embed_dim" : 768,  # Embedding size for each token
        "num_heads" : 12,  # Number of attention heads
        "num_layers": 12,
        "feed_forward_dim": 1024  # Hidden layer size in feed forward network inside transformer
    }

    batch_size = 128


    model_name = f"transformer_h{hparams['num_heads']}l{hparams['num_layers']}emb{hparams['embed_dim']}"
    model_folder = "models/" + model_name
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    with codecs.open (model_folder + "/params.json", "w", "utf-8") as f:
        f.write(json.dumps(hparams))
    model = TransformerModel(hparams)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=300,
        decay_rate=0.96)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer, loss=[loss_fn, None],
    )  
    X_test, _, _, _ = prepare_data.getTestTrain(hparams["maxlen"], 0.05)
    pred_test = model(X_test[:10])
    print("pred_test shape", pred_test.shape)
    print(model.summary())


    text_gen_callback = TextGenerator(hparams["maxlen"], 28, "And she said", file=model_folder + "/generated_samples.txt")


    checkpoint_path = f"models/{model_name}/ckpt.ckpt"

    start_over = args.start_over

    if Path(checkpoint_path + ".index").exists() and (not start_over):
        print("loading existing model")
        model.load_weights(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    epochs = 200

    for epoch in range(1, epochs + 1):
        print("epoch: " , epoch)
        X_test, Y_test, X_train, Y_train = prepare_data.getTestTrain(hparams["maxlen"], 0.05)
        model.fit(X_train[:50*batch_size], Y_train[:50*batch_size], verbose=1, epochs=1, batch_size=batch_size, callbacks=[text_gen_callback, cp_callback])