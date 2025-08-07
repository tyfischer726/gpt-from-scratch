import os
import numpy as np
import tokenizer

output_dir = 'training_outputs'

with open('datasets/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print('\nEncoding data...')
data = tokenizer.encode(text)

if output_dir not in os.listdir():
    os.mkdir(output_dir)
data = np.array(data)
np.save(f'{output_dir}/data.npy', data)

print(f'len text:\t\t{len(text)}')
print(f'len data:\t\t{len(data)}')

print('\nDone.\n')