"""
包含基础 Tokenizer 类和一些通用的辅助函数。
基类还包含了（通用的）保存/加载功能。
在严格接口隔离方面，本可以将所有正则/模式相关部分
完全隔离到 RegexTokenizer 中，但为了简单起见做了一些让步。
"""
import unicodedata

# -----------------------------------------------------------------------------
# 一些对 BasicTokenizer 和 RegexTokenizer 都有用的辅助函数

def get_stats(ids, counts=None):
    """
    给定一个整数列表，返回连续对出现次数的字典
    示例: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    可选参数允许更新已有的计数字典
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # 遍历连续元素
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    在整数列表 (ids) 中，将所有连续出现的 pair
    替换为新的整数 token idx
    示例: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # 如果不在最后一个位置，且 pair 匹配，则替换它
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# 另外两个辅助函数...
def replace_control_characters(s: str) -> str:
    # 我们不想打印控制字符
    # 它们会扭曲输出（例如 \n 或更糟的情况）
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # 这个字符没问题
        else:
            chars.append(f"\\u{ord(ch):04x}") # 转义
    return "".join(chars)

def render_token(t: bytes) -> str:
    # 美化打印 token，转义控制字符
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# -----------------------------------------------------------------------------
# 基础 Tokenizer 类

class Tokenizer:
    """分词器的基类"""

    def __init__(self):
        # 默认：词汇表大小为 256（所有字节），无合并，无模式
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, 例如 {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # 分词器可以从文本中训练一个大小为 vocab_size 的词汇表
        raise NotImplementedError

    def encode(self, text):
        # 分词器可以将字符串编码为整数列表
        raise NotImplementedError

    def decode(self, ids):
        # 分词器可以将整数列表解码为字符串
        raise NotImplementedError

    def _build_vocab(self):
        # vocab 简单而确定地从 merges 派生而来
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        保存两个文件：file_prefix.vocab 和 file_prefix.model
        这是受 sentencepiece 的模型保存方式的启发（但并不等同于它）：
        - model 文件是关键文件，用于 load()
        - vocab 文件只是格式化打印版本，仅供人类查看
        """
        # 写入 model 文件：供 load() 后续使用
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # 写入版本、模式和合并表，这就是全部所需信息
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # 写入特殊 token，首先是数量，然后是每个 token
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # merges 字典
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # 写入 vocab 文件：供人类查看
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # 注意：很多 token 可能是部分 utf-8 序列
                # 无法解码为有效字符串。这里我们使用
                # errors='replace' 将其替换为替换字符 �。
                # 这也意味着我们不能在 load() 中使用 .vocab 文件
                # 因为这种解码方式是有损操作！
                s = render_token(token)
                # 找到此 token 的子 token（如果有的话）
                if idx in inverted_merges:
                    # 如果此 token 有子 token，美化渲染为合并形式
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # 否则这是叶子 token，直接打印
                    # （这应该只是前 256 个 token，即字节）
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """save() 的反操作，但仅针对 model 文件"""
        assert model_file.endswith(".model")
        # 读取 model 文件
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # 读取版本
            version = f.readline().strip()
            assert version == "minbpe v1"
            # 读取模式
            self.pattern = f.readline().strip()
            # 读取特殊 token
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # 读取合并表
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
