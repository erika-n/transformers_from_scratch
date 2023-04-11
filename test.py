import miniature
import json
from transformers import AutoTokenizer
import tensorflow as tf
import numpy as np
import prepare_data
import matplotlib.pyplot as plt

num_heads = 1
num_layers = 1
embed_dim = 256

model_name = f"transformer_h{num_heads}l{num_layers}emb{embed_dim}"

model_folder = "models/" + model_name

print("model folder: ", model_folder)

with open(model_folder + "/params.json") as f:
    hparams = json.loads(f.read())
checkpoint_path = f"models/{model_name}/ckpt.ckpt"

the_model = miniature.TransformerModel(hparams)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
the_model.compile(
    "adam", loss=[loss_fn, None],
)  
X_test, _, _, _ = prepare_data.getTestTrain(hparams["maxlen"], 0.05)
pred_test = the_model(X_test[:10])
print("pred_test shape", pred_test.shape)
print(the_model.summary())


the_model.load_weights(checkpoint_path)


prompt = "The bear went over the mountain, the bear went over the mountain, the bear went over the mountain to see what he could see. The other side of the mountain, the other side of the mountain, the other side of the mountain was all that he could see."
tokenizer = AutoTokenizer.from_pretrained("gpt2")
start_tokens = tokenizer.encode(prompt)
print("n tokens", len(start_tokens))
pad_len = hparams["maxlen"] - len(start_tokens)
x = np.array(start_tokens + [0] * pad_len)
print("x", x.shape)
x = x.reshape((1, -1))
preds = the_model(x)
print("preds", preds.shape)
tokens = tf.math.argmax(preds, 2)
tokens 
print("tokens", tokens.shape)

tokens = tf.reshape(tokens, (80, 1))
tokens = tokens.numpy()
print(tokens)
decoded = tokenizer.batch_decode(tokens)
#decoded.insert(0, "")
print("decoded: ", decoded)



w = the_model.transformers.layers[0].att.w
print("eager execution: ", tf.executing_eagerly())
print("w", w.numpy())
fig, ax = plt.subplots()
w = tf.squeeze(w)
show_len = 6
text = tokenizer.decode(start_tokens[:show_len])
text = "intro " + text
text = text.split(" ")
print(text)

axr = ax.twinx()
axr.set_box_aspect(1)
axr.set_ylabel("Output")
ax.set_xticklabels(text)
ax.set_yticklabels(text)
ax.yaxis.tick_left()

axr.set_ylim(0, 6)
axr.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
axr.yaxis.tick_right()
axr.set_yticklabels(decoded[:show_len][::-1])
ax.set_xlabel("Keys")
ax.set_ylabel("Queries")
ax.imshow(w[:show_len, :show_len])
plt.title("Attention Head in Action")
plt.show()