import miniature
import json
from transformers import AutoTokenizer
import tensorflow as tf
import numpy as np

num_heads = 2
num_layers = 2
embed_dim = 256

model_name = f"test_h{num_heads}l{num_layers}emb{embed_dim}"

model_folder = "models/" + model_name

with open(model_folder + "/params.json") as f:
    hparams = json.loads(f.read())
checkpoint_path = f"models/{model_name}/ckpt.ckpt"

the_model = miniature.create_model(hparams)
the_model.load_weights(checkpoint_path)


prompt = "The bear went over the mountain to see what he could see."
tokenizer = AutoTokenizer.from_pretrained("gpt2")
start_tokens = tokenizer.encode(prompt)
pad_len = hparams["maxlen"] - len(start_tokens)
x = np.array(start_tokens + [0] * pad_len)
print("run_model")
print(x.shape)
x = x.reshape((1, -1))
y = the_model.predict(x, verbose=0)
print(y.shape)
print(the_model.transformer_blocks.shape)