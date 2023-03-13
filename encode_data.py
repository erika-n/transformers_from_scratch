from transformers import AutoTokenizer
from os import listdir
import numpy as np
import codecs

# tokenize and encode great books dataset, saving as encoded_tokens.npy

tokenizer = AutoTokenizer.from_pretrained("gpt2")

booksdir = '../text/library/great_books_program'
folders = [f for f in listdir(booksdir)]
files = []

for folder in folders:
    these_files = [booksdir + '/' + folder + '/' + f for f in listdir(booksdir +'/'+ folder)]
    files.extend(these_files)


all_encoded_tokens = np.array([])

for textfile in files:
     
    with codecs.open(textfile, "r", "utf-8") as f:
        text = f.read()
    tokens = tokenizer.tokenize(text)
    encoded_tokens = tokenizer.convert_tokens_to_ids(tokens)
    print(textfile)
  
    all_encoded_tokens = np.concatenate((all_encoded_tokens, np.array(encoded_tokens)))

print(all_encoded_tokens[:10])
print(all_encoded_tokens.shape)
print("max: ", np.max(all_encoded_tokens    ))
with open('encoded_tokens.npy', 'wb') as f:
    np.save(f, all_encoded_tokens)


