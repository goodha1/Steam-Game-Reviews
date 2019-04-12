import re
# import string
import utils
from collections import defaultdict
""" Byte-Pair Encoding
    Of course in real algorithms BPE is run with many thousands of merges on a very large input dictionary.
    When we need to tokenize a test sentence, we just run the merges we have learned, greedily, in the order we learned them,
"""


def bpe_symbolize(word):
    return " ".join(list(word)) + " </w>"


def count_pairs(bpe_v):
    pairs = defaultdict(int)
    for word, freq in bpe_v.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v):
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v[word]
    return v_out


def BPE(num_merge, vocab):
    bpe_v = {bpe_symbolize(k): vocab[k] for k in vocab.keys()}
    learned_mearge = []
    for i in range(num_merge):
        pairs = count_pairs(bpe_v)
        best = max(pairs, key=pairs.get)
        bpe_v = merge_vocab(best, bpe_v)
        learned_mearge.append(best)
    return bpe_v, learned_mearge


def merge_bpe_symbol(sym, pairs):
    for pair in pairs:
        bigram = re.escape(" ".join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        sym = p.sub(''.join(pair), sym)
    return sym


def desymbolize(sym):
    return sym.replace("</w>", "").strip()


def _tokenize(pairs, fin, fout):
    out = open(fout, "w")
    out.write("label\treview\n")
    with open(fin) as f:
        f.readline()
        for line in f:
            rec = line.strip().split()
            syms = [bpe_symbolize(utils._word(token)) for token in rec[1:]]
            bpe_word_syms = [merge_bpe_symbol(sym, pairs) for sym in syms]
            out.write("\t".join([rec[0]] + [desymbolize(s) for s in bpe_word_syms]) + "\n")

    out.close()


if __name__ == "__main__":
    print("Constructing vocabulary...")
    raw_v = utils._vocab("review-ascii-only.dev")
    print('Done!\n')
    print('Running Byte-Pair Encoding(BPE algorithm) and learn words stemming pairs... ')
    bpe_v, learned_mearge = BPE(3000, raw_v)
    print('Done!\n')
    print("Re-tokenizing gaming reviews with merges leaned in BPE...")
    _tokenize(learned_mearge, "review-ascii-only.dev", "review-ascii-only.dev.bpe_tokenized")
    print("Done!")
