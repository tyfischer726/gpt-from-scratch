with open('datasets/taylorswift.txt', 'r', encoding='utf-8') as f:
    tok_text = f.read()

utf8_tokens = list(tok_text.encode('utf-8'))

def get_stats(tokens):
    stats = {}
    for pair in zip(tokens, tokens[1:]):
        stats[pair] = stats.get(pair, 0) + 1
    return stats

def merge(tokens, pair, idx):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if (i < len(tokens)-1) and (tokens[i]==pair[0] and tokens[i+1]==pair[1]):
            new_tokens.append(idx)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

num_merges = 100
tokens = list(utf8_tokens)
merges = {}
print('\nMerging tokenizer data...')
for i in range(num_merges):
    stats = get_stats(tokens)
    pair = max(stats, key=stats.get)
    if stats[pair] == 1:
        break
    idx = 256 + i
    tokens = merge(tokens, pair, idx)
    merges[pair] = idx
    # print(f'merged\t{pair}\tinto\t{idx}')
print(f'Done merging.')

vocab = {idx : bytes([idx]) for idx in range(256)}
for pair in merges:
    idx = merges[pair]
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
vocab_size = len(vocab)

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

print('\nVerifying tokenizer:')
test = "Hello, world."
print(f'test:\t\t{test}')
print(f"utf8:\t\t{list(test.encode('utf-8'))}")

test = encode(test)
print(f'encoded:\t{test}')

test = decode(test)
print(f'decoded:\t{test}')