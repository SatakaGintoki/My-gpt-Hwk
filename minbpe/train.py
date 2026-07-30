"""
在测试数据上训练我们的分词器，看看它们实际运行的效果。
整个过程在我的笔记本上大约运行 25 秒。
"""

import os
import time
from minbpe import BasicTokenizer, RegexTokenizer

# 打开一些文本，训练一个 512 个 token 的词汇表
text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()

# 为模型创建一个目录，以免污染当前目录
os.makedirs("models", exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):

    # 构造分词器对象并启动详细训练
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)
    # 在 models 目录中写入两个文件：name.model 和 name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.time()

print(f"训练耗时 {t1 - t0:.2f} 秒")
