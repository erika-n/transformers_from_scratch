import miniature
import json
from transformers import AutoTokenizer
import tensorflow as tf
import numpy as np
import prepare_data
import matplotlib.pyplot as plt
import matplotlib


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "The shepherd watched over his flock"
    
    # which model to load
    num_heads = 4
    num_layers = 2
    embed_dim = 512
    tag = "transformer"
    model, hparams = load_model(num_heads,num_layers, embed_dim, tag)
    

    for layer in range(num_layers):
        
        for head in range(num_heads):
            fig, ax = plt.subplots()
            w, input_labels, output_labels, value_labels = getMatrix(model, tokenizer, hparams, prompt,head, layer)
            print(f"w.shape: {w.shape}")
            print(f"input_labels: {input_labels}")
            print(f"output_labels: {output_labels}")
            print(f"value_labels: {value_labels}")
           
            print(f"creating subplot for layer: {layer}, head: {head}")
            createSubplot(w, ax, input_labels, output_labels, value_labels)
            ax.set_title(f"Layer {layer}, Head {head}")

            plt.tight_layout() 
            plt.savefig(f"images/{tag}_h{num_heads}l{num_layers}emb{embed_dim}_l{layer}h{head}.jpg")
            plt.show()
            plt.clf()
    

def load_model(num_heads, num_layers, embed_dim, tag):


    model_name = f"{tag}_h{num_heads}l{num_layers}emb{embed_dim}"

    model_folder = "models/" + model_name

    print("model folder: ", model_folder)

    # load and test model
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


    return the_model, hparams

# get attention matrix for given head and layer
def getMatrix(model, tokenizer, hparams, prompt,head, layer):

    # get input
    input_tokens = tokenizer.encode(prompt)
   
    pad_len = hparams["maxlen"] - len(input_tokens)
    input_tokens = np.array(input_tokens + [0] * pad_len)
    
    input_tokens = input_tokens.reshape((1, -1))

    # make prediction
    preds = model(input_tokens, use_layer=layer, use_head=head, values_only=False)
    # get attention matrix for given layer and head
    w = model.blocks.layers[layer].att.heads[head].w[:]

    # get output tokens
    tokens = tf.math.argmax(preds, 2)
    tokens = tf.reshape(tokens, (hparams["maxlen"], 1))
    tokens = tokens.numpy()

    # decode output tokens
    output_labels = tokenizer.batch_decode(tokens)

    # get value only output tokens
    preds = model(input_tokens, use_layer=layer, use_head=head, values_only=True)    
    tokens = tf.math.argmax(preds, 2)
    tokens = tf.reshape(tokens, (hparams["maxlen"], 1))
    tokens = tokens.numpy()
    value_labels = tokenizer.batch_decode(tokens)


    # get input labels
    input_tokens = tf.reshape(input_tokens, (hparams["maxlen"], 1))
    input_tokens = input_tokens.numpy()
    input_labels = tokenizer.batch_decode(input_tokens)


    return w, input_labels, output_labels, value_labels



def createSubplot(w, ax, input_labels, output_labels, value_labels, show_len = 6):


    w = tf.squeeze(w)
    input_labels = input_labels[:show_len]
    input_labels.insert(0, ".")
    

    axr = ax.twinx()
    axr.set_box_aspect(1)
    axr.set_ylabel("Output")

    axu = ax.twiny()
    axu.set_box_aspect(1)
    axu.set_xlabel("Values")

    ax.set_xticklabels(input_labels, rotation=45)
    ax.set_yticklabels(input_labels)
    ax.yaxis.tick_left()

    axr.set_ylim(0, show_len)
    axr.set_yticks(np.arange(0.5, show_len + 0.5, 1))
    axr.yaxis.tick_right()


    axu.set_xlim(0, show_len)
    axu.set_xticks(np.arange(0.5, show_len + 0.5, 1))

    #output_labels = [output_labels[i] + " (" + value_labels[i] + ")" for i in range(len(output_labels))]
    print(f"output_labels for imshow {output_labels[:show_len][::-1]}")
    axr.set_yticklabels(output_labels[:show_len][::-1])
    axu.set_xticklabels(value_labels[:show_len], rotation=45)
    
    ax.set_xlabel("Keys")
    ax.set_ylabel("Queries")
    print(f"for imshow, w.shape: {w.shape}")
    ax.imshow(w[:show_len, :show_len])


if __name__ == "__main__":
    main()
