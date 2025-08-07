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

if __name__ == '__main__':
    import json

    with open('datasets/taylorswift.txt', 'r', encoding='utf-8') as f:
        tok_text = f.read()

    utf8_tokens = list(tok_text.encode('utf-8'))

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
        print(f'merged\t{pair}\tinto\t{idx}')
    print(f'Done merging.')

    vocab = {idx : bytes([idx]) for idx in range(256)}
    for pair in merges:
        idx = merges[pair]
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    vocab_size = len(vocab)

    vocab_out = {}
    for idx in vocab:
        vocab_out[idx] = list(vocab[idx])
    with open('vocab.json', 'w') as f:
        json.dump(vocab_out, f, indent=4)
    print('\nSaved vocab to vocab.json')

    merges_out = {}
    for pair in merges:
        pair_str = str(pair)
        merges_out[pair_str] = merges[pair]
    with open('merges.json', 'w') as f:
        json.dump(merges_out, f, indent=4)
    print('Saved merges to merges.json')
    print('')