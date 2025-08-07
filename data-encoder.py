import tokenizer
import numpy as np

with open('datasets/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print('\nEncoding data...')
data = tokenizer.encode(text)
data = np.array(data)
np.save('data.npy', data)

print(f'len text:\t\t{len(text)}')
print(f'len data:\t\t{len(data)}')

print('\nDone.\n')