"""
BPE 训练函数完整演示 —— 仅用于学习对照，不修改你的作业代码。

跑一下这个文件，看中间输出，就能理解每一步数据长什么样：
    python d:/File/for/Study/CS336/train_bpe_demo.py
"""

import regex as re


# ============================================================
# 工具函数（和你的 merge / get_state 一样）
# ============================================================

def merge(ids, pair, idx):
    """把 ids 里所有相邻的 pair 替换为 idx"""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def get_stats(ids, counts):
    """统计连续对出现次数，结果累加到 counts 里"""
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# ============================================================
# 训练函数
# ============================================================

def train_bpe(input_path, vocab_size, special_tokens=None):
    if special_tokens is None:
        special_tokens = []

    # ----------------------------------------
    # Step 1: 初始化词汇表（0~255 是原始字节）
    # ----------------------------------------
    vocab = {idx: bytes([idx]) for idx in range(256)}
    merges = []

    # ----------------------------------------
    # Step 2: 分配特殊 token 的 ID
    # ----------------------------------------
    next_id = 256
    special_ids = {}  # 存特殊 token → ID 的映射，后面训练时用不到，但返回有用
    for st in special_tokens:
        special_ids[st] = next_id
        vocab[next_id] = st.encode("utf-8")
        next_id += 1

    print(f"特殊 token 已分配: {special_ids}")
    print(f"下一个可用 ID: {next_id}")

    # ----------------------------------------
    # Step 3: 读取文本
    # ----------------------------------------
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # ----------------------------------------
    # Step 4: 用 re.split 剥离特殊 token，然后分别处理
    # ----------------------------------------
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    text_byte = []

    if special_tokens:
        # 拼出分割用的正则，例如 "(<\|endoftext\|>|<\|pad\|>)"
        escaped = [re.escape(t) for t in special_tokens]
        special_pattern = "(" + "|".join(escaped) + ")"
        # 分割
        parts = re.split(special_pattern, text)
        # 分类处理
        for part in parts:
            if part in special_ids:          # <-- 是特殊 token
                continue                      #     跳过，不参与合并统计
            elif part == "":                 # <-- 空字符串（两个相邻特殊 token 之间）
                continue                      #     也跳过
            else:                            # <-- 普通文本
                for chunk in re.findall(pattern, part):
                    text_byte.append(list(chunk.encode("utf-8")))
    else:
        # 没有特殊 token，直接 findall
        for chunk in re.findall(pattern, text):
            text_byte.append(list(chunk.encode("utf-8")))

    print(f"text_byte 共有 {len(text_byte)} 个 chunk")
    print(f"前 3 个 chunk 示例: {text_byte[:3]}")

    # ----------------------------------------
    # Step 5: BPE 合并循环
    # ----------------------------------------
    num_merges = vocab_size - len(vocab)  # 还剩多少空位

    for i in range(num_merges):
        # 统计
        counts = {}
        for word in text_byte:
            get_stats(word, counts)

        if not counts:
            print(f"第 {i} 轮没有相邻对了，提前结束")
            break

        # 找出现最多的 pair（字典序平局自动处理）
        max_pair = max(counts, key=counts.get)

        # 合并
        text_byte = [merge(chunk, max_pair, next_id) for chunk in text_byte]

        # 记录
        vocab[next_id] = vocab[max_pair[0]] + vocab[max_pair[1]]
        merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))

        if i < 3 or i % 10 == 0:
            print(f"  合并 {i+1}/{num_merges}: "
                  f"pair={max_pair} ({vocab[max_pair[0]]!r} + {vocab[max_pair[1]]!r}) "
                  f"→ id={next_id} ({vocab[next_id]!r}), 出现 {counts[max_pair]} 次")

        next_id += 1

    return vocab, merges


# ============================================================
# 跑一下看看
# ============================================================
if __name__ == "__main__":
    # 用测试自带的语料
    import pathlib
    corpus_path = pathlib.Path("d:/File for Study/CS336/assignment1-basics/tests/fixtures/corpus.en")

    vocab, merges = train_bpe(
        input_path=str(corpus_path),
        vocab_size=300,                      # 小一点方便看
        special_tokens=["<|endoftext|>"],
    )

    print(f"\n最终 vocab 大小: {len(vocab)}")
    print(f"合并次数: {len(merges)}")
    print(f"前 5 个 merge: {merges[:5]}")
