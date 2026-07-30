# 练习

从零构建你自己的 GPT-4 分词器（Tokenizer）！

### 第一步

编写 `BasicTokenizer` 类，实现以下三个核心函数：

- `def train(self, text, vocab_size, verbose=False)`
- `def encode(self, text)`
- `def decode(self, ids)`

用任意文本训练你的分词器，并可视化合并后的 token。它们看起来合理吗？你可以用 `tests/taylorswift.txt` 文件作为默认的测试文本。

### 第二步

将你的 `BasicTokenizer` 改造为 `RegexTokenizer`，它接受一个正则表达式模式，并按照 GPT-4 的方式分割文本。分别处理每个分割后的部分（与之前相同），然后将结果拼接起来。重新训练你的分词器，并比较之前和之后的结果。你会发现现在不会有跨类别（数字、字母、标点、多于一个空格）的 token 出现了。使用 GPT-4 的分割模式：

```
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

### 第三步

现在你已经准备好加载 GPT-4 分词器的合并表，并证明你的分词器在 `encode` 和 `decode` 上都能产生与 [tiktoken](https://github.com/openai/tiktoken) 完全相同的结果。

```
# 匹配以下结果
import tiktoken
enc = tiktoken.get_encoding("cl100k_base") # 这是 GPT-4 的分词器
ids = enc.encode("hello world!!!? (안녕하세요!) lol123 😉")
text = enc.decode(ids) # 能得到相同的原始文本
```

不过，你会遇到两个问题：

1. 从 GPT-4 分词器中恢复原始合并表并不简单。你可以轻松地恢复我们这里称为 `vocab`（词汇表）的东西，也就是他们在 `enc._mergeable_ranks` 中存储的内容。你可以直接复制粘贴 `minbpe/gpt4.py` 中的 `recover_merges` 函数，它接受这些排名并返回原始的合并表。如果你想了解这个函数的工作原理，请阅读[这里](https://github.com/openai/tiktoken/issues/60)和[这里](https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306)。简单来说，在某些条件下，只需要存储父节点（及其排名），而不需要知道具体是哪两个子节点合并成了某个父节点。
2. 其次，GPT-4 分词器出于某种原因会对其原始字节进行置换。它将这个置换存储在前 256 个合并排名元素中，因此你可以相对简单地恢复这个字节重排映射：`byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(256)}`。在你的 `encode` 和 `decode` 中，都需要相应地重排字节。如果你卡住了，可以参考 `minbpe/gpt4.py` 文件中的提示。

### 第四步

（可选，繁琐，且看起来没什么明显用处）添加处理特殊 token 的能力。这样即使存在特殊 token，你也能匹配 tiktoken 的输出，例如：

```
import tiktoken
enc = tiktoken.get_encoding("cl100k_base") # 这是 GPT-4 的分词器
ids = enc.encode("<|endoftext|>hello world", allowed_special="all")
```

如果不加 `allowed_special` 参数，tiktoken 会报错。

### 第五步

如果你已经走到了这一步，恭喜你——你现在已经是 LLM 分词领域的专家了！不过遗憾的是，你的旅程还没有完全结束，因为 OpenAI 之外的很多 LLM（例如 Llama、Mistral）使用的是 [sentencepiece](https://github.com/google/sentencepiece)。主要区别在于 sentencepiece 直接在 Unicode 码点上运行 BPE，而不是在 UTF-8 编码的字节上运行。如果你有兴趣，可以自行探索 sentencepiece（祝你好运，它的代码不太好看）。作为一个高难度的额外挑战——如果你真的有很多时间并且愿意受苦——可以把你的 BPE 改写为基于 Unicode 码点的版本，并匹配 Llama 2 的分词器。
