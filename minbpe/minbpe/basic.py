"""
最小化的（字节级）Byte Pair Encoding 分词器。

算法上遵循 GPT 分词器：
https://github.com/openai/gpt-2/blob/master/src/encoder.py

但有以下区别：
- 不处理正则表达式分割模式。
- 不处理任何特殊 token。
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 输入文本预处理
        text_bytes = text.encode("utf-8") # 原始字节
        ids = list(text_bytes) # 范围在 0..255 的整数列表

        # 迭代合并最常出现的对，以创建新 token
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # 统计每个连续对出现的次数
            stats = get_stats(ids)
            # 找到出现次数最多的对
            pair = max(stats, key=stats.get)
            # 铸造一个新 token：分配下一个可用的 id
            idx = 256 + i
            # 将 ids 中所有出现的 pair 替换为 idx
            ids = merge(ids, pair, idx)
            # 保存合并
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # 打印
            if verbose:
                print(f"合并 {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) 出现了 {stats[pair]} 次")

        # 保存类变量
        self.merges = merges # 用于 encode()
        self.vocab = vocab   # 用于 decode()

    def decode(self, ids):
        # 给定 ids（整数列表），返回 Python 字符串
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # 给定字符串 text，返回 token id 列表
        text_bytes = text.encode("utf-8") # 原始字节
        ids = list(text_bytes) # 范围在 0..255 的整数列表
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
