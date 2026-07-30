"""
最小化的（字节级）Byte Pair Encoding 分词器。

算法上遵循 GPT 分词器：
https://github.com/openai/gpt-2/blob/master/src/encoder.py

与 BasicTokenizer 不同：
- RegexTokenizer 处理可选的正则分割模式。
- RegexTokenizer 处理可选的特殊 token。
"""

import regex as re
from .base import Tokenizer, get_stats, merge


# GPT 主要的文本分割模式，参见
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: 可选的字符串，用于覆盖默认模式（GPT-4 分割模式）
        - special_tokens: str -> int 的特殊 token 字典
          示例: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 将文本分割为文本块
        text_chunks = re.findall(self.compiled_pattern, text)

        # 输入文本预处理
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # 迭代合并最常出现的对，以创建新 token
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # 统计每个连续对出现的次数
            stats = {}
            for chunk_ids in ids:
                # 传入 stats 会原地更新它，累加计数
                get_stats(chunk_ids, stats)
            # 找到出现次数最多的对
            pair = max(stats, key=stats.get)
            # 铸造一个新 token：分配下一个可用的 id
            idx = 256 + i
            # 将 ids 中所有 chunk 里出现的 pair 替换为 idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # 保存合并
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # 打印
            if verbose:
                print(f"合并 {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) 出现了 {stats[pair]} 次")

        # 保存类变量
        self.merges = merges # 用于 encode()
        self.vocab = vocab   # 用于 decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens 是一个 str -> int 的字典
        # 示例: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # 给定 ids（整数列表），返回 Python 字符串
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"无效的 token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # 返回 token id
        # 开始。首先，将所有字节转换为 0..255 范围内的整数
        ids = list(text_bytes)
        while len(ids) >= 2:
            # 找到合并索引最小的对
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # 微妙之处：如果没有更多可用的合并，min 的 key 对于
            # 每一对都会返回 inf，min 将任意选择列表中的第一对
            # 我们可以通过成员检查来检测这种终止条件
            if pair not in self.merges:
                break # 没有更多的合并可以做
            # 否则合并最佳的对（合并索引最小的）
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """忽略任何特殊 token 的编码。"""
        # 按照正则模式定义的类别将文本分割为文本块
        text_chunks = re.findall(self.compiled_pattern, text)
        # 所有文本块分别编码，然后拼接结果
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # 原始字节
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        与 encode_ordinary 不同，此函数处理特殊 token。
        allowed_special: 可以是 "all"|"none"|"none_raise" 或一个自定义的特殊 token 集合
        如果是 none_raise，则当文本中遇到任何特殊 token 时会抛出错误
        这也是 tiktoken 当前的默认行为
        其他任何行为要么令人烦恼，要么是一个巨大的隐患
        """
        # 解码用户对特殊 token 处理的意图
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} 无法理解")
        if not special:
            # 快捷路径：如果没有特殊 token，直接使用普通编码
            return self.encode_ordinary(text)
        # 否则，我们需要小心处理文本中可能出现的特殊 token
        # 我们通过分割文本的方式来处理特殊 token
        # 基于与任何特殊 token 的精确匹配
        # 我们可以使用 re.split 来做这件事。注意用 () 包围模式
        # 使其成为捕获组，这样特殊 token 也会被包含在结果中
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # 现在所有特殊字符都从文本其余部分中分离出来了
        # 所有文本块分别编码，然后拼接结果
        ids = []
        for part in special_chunks:
            if part in special:
                # 这是一个特殊 token，作为特殊情况单独编码
                ids.append(special[part])
            else:
                # 这是一个普通序列，正常编码
                ids.extend(self.encode_ordinary(part))
        return ids
