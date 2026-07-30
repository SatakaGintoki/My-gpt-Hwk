"""
实现 GPT-4 分词器，作为 RegexTokenizer 的轻量封装。
注意这是一个预训练的分词器。默认情况下，在 __init__() 中，
它会从 tiktoken 的 `cl100k_base` 分词器加载预训练的权重。
"""

import tiktoken
from .regex import RegexTokenizer


def bpe(mergeable_ranks, token, max_rank):
    # 辅助函数，在 get_gpt4_merges() 中用于重建合并森林
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def recover_merges(mergeable_ranks):
    # `merges` 中已经包含了合并后的字节序列。
    # 所以我们必须恢复原始的配对关系。我们可以通过对
    # 所有 token 按顺序执行一次小型 BPE 训练来实现这一点。
    # 另见 https://github.com/openai/tiktoken/issues/60
    # 另见 https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # 跳过原始字节
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # 恢复该对的整数排名
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class GPT4Tokenizer(RegexTokenizer):
    """对 RegexTokenizer 的轻量封装，匹配 GPT-4 的分词器。"""

    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        # 获取官方分词器及其合并表
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        # 这些是 gpt4 的合并，但我们必须恢复它们
        self.merges = recover_merges(mergeable_ranks)
        # 从合并表中重建 vocab
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        # 现在这里是另一个棘手的问题。
        # 出于某种原因，单个字节对应的 token
        # 被以不同的顺序排列了。这完全说不通
        # 而且很可能是历史遗留问题，但我们不得不在这里处理它。
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        # 最后注册特殊 token
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, text_bytes):
        # 在开始处理字节之前，我们必须对它们进行置换
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids = super()._encode_chunk(text_bytes)
        return ids

    def decode(self, ids):
        # 在解码之前，我们必须对字节进行反向置换
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    # 这是一个预训练的分词器，不打算被训练
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    # 保存/加载需要一些思考。
    # 我们不得不修改基类的 save/load 以添加对 byte_shuffle 的支持...
    # 或者，我们可以把 byte_shuffle 移到基类，但那意味着
    # 仅仅为了支持 GPT-4 分词器及其奇怪的
    # byte_shuffle 历史遗留问题，就要把我们漂亮的 Tokenizer 搞得很难看。
    def save(self, file_prefix):
        raise NotImplementedError("GPT4Tokenizer 不能被保存。")

    def load(self, model_file):
        raise NotImplementedError("GPT4Tokenizer 不能被加载。")

    def save_vocab(self, vocab_file):
        # 仅仅为了可视化目的，让我们以和基类完全相同的格式
        # 输出 GPT-4 的 token。
        # 简单用法：
        # python -c "from minbpe import GPT4Tokenizer; GPT4Tokenizer().save_vocab('gpt4.vocab')"
        from .base import render_token
        # 构建 vocab，注意字节重排
        vocab = {idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # 现在合并重排后的字节并写入文件
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")
