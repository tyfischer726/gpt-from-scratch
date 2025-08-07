import json
import ast
from tokenizertrain import get_stats, merge

with open('merges.json', 'r') as f:
    merges_json = json.load(f)
with open('vocab.json', 'r') as f:
    vocab_json = json.load(f)

merges = {}
for key_json in merges_json:
    key = ast.literal_eval(key_json)
    merges[key] = merges_json[key_json]

vocab = {}
for key_json, value_json in vocab_json.items():
    key = ast.literal_eval(key_json)
    value = b''.join([bytes([i]) for i in value_json])
    vocab[key] = value

def encode(text):
    tokens = list(text.encode('utf-8'))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

def decode(tokens):
    text = b''.join([vocab[t] for t in tokens])
    text = text.decode('utf-8', errors='replace')
    return text

if __name__ == "__main__":
    print('\nVerifying tokenizer:')
    test = "Hello, world."
    print(f'test:\t\t{test}')
    print(f"utf8:\t\t{list(test.encode('utf-8'))}")

    test = encode(test)
    print(f'encoded:\t{test}')

    test = decode(test)
    print(f'decoded:\t{test}\n')