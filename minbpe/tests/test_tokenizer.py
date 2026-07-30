import pytest
import tiktoken
import os

from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

# -----------------------------------------------------------------------------
# 通用测试数据

# 用于测试分词器的几个字符串
test_strings = [
    "", # 空字符串
    "?", # 单个字符
    "hello world!!!? (안녕하세요!) lol123 😉", # 有趣的小字符串
    "FILE:taylorswift.txt", # FILE: 在 unpack() 中被作为特殊字符串处理
]
def unpack(text):
    # 这样做是因为 `pytest -v .` 会将参数打印到控制台，
    # 而我们不想打印文件的全部内容，那会造成混乱。所以在这里处理。
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        taylorswift_file = os.path.join(dirname, text[5:])
        contents = open(taylorswift_file, "r", encoding="utf-8").read()
        return contents
    else:
        return text

specials_string = """
<|endoftext|>Hello world this is one document
<|endoftext|>And this is another document
<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
<|endoftext|>Last document!!! 👋<|endofprompt|>
""".strip()
special_tokens = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}
llama_text = """
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
""".strip()

# -----------------------------------------------------------------------------
# 测试用例

# 测试几个不同字符串的 encode/decode 恒等性
@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer, RegexTokenizer, GPT4Tokenizer])
@pytest.mark.parametrize("text", test_strings)
def test_encode_decode_identity(tokenizer_factory, text):
    text = unpack(text)
    tokenizer = tokenizer_factory()
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert text == decoded

# 测试我们的分词器是否匹配官方的 GPT-4 分词器
@pytest.mark.parametrize("text", test_strings)
def test_gpt4_tiktoken_equality(text):
    text = unpack(text)
    tokenizer = GPT4Tokenizer()
    enc = tiktoken.get_encoding("cl100k_base")
    tiktoken_ids = enc.encode(text)
    gpt4_tokenizer_ids = tokenizer.encode(text)
    assert gpt4_tokenizer_ids == tiktoken_ids

# 测试特殊 token 的处理
def test_gpt4_tiktoken_equality_special_tokens():
    tokenizer = GPT4Tokenizer()
    enc = tiktoken.get_encoding("cl100k_base")
    tiktoken_ids = enc.encode(specials_string, allowed_special="all")
    gpt4_tokenizer_ids = tokenizer.encode(specials_string, allowed_special="all")
    assert gpt4_tokenizer_ids == tiktoken_ids

# 参考测试，以后可以添加更多测试
@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer, RegexTokenizer])
def test_wikipedia_example(tokenizer_factory):
    """
    快速单元测试，跟随维基百科的示例：
    https://en.wikipedia.org/wiki/Byte_pair_encoding

    根据维基百科，对输入字符串：
    "aaabdaaabac"

    进行 3 次合并会得到：
    "XdXac"

    其中：
    X=ZY
    Y=ab
    Z=aa

    请记住，对我们来说 a=97, b=98, c=99, d=100（ASCII 值）
    所以 Z 将是 256，Y 将是 257，X 将是 258。

    因此我们期望输出 id 列表为 [258, 100, 258, 97, 99]
    """
    tokenizer = tokenizer_factory()
    text = "aaabdaaabac"
    tokenizer.train(text, 256 + 3)
    ids = tokenizer.encode(text)
    assert ids == [258, 100, 258, 97, 99]
    assert tokenizer.decode(tokenizer.encode(text)) == text

@pytest.mark.parametrize("special_tokens", [{}, special_tokens])
def test_save_load(special_tokens):
    # 取一段稍微复杂一点的文本来训练分词器，随机选取的
    text = llama_text
    # 创建一个分词器并做 64 次合并
    tokenizer = RegexTokenizer()
    tokenizer.train(text, 256 + 64)
    tokenizer.register_special_tokens(special_tokens)
    # 验证 decode(encode(x)) == x
    assert tokenizer.decode(tokenizer.encode(text, "all")) == text
    # 验证 save/load 按预期工作
    ids = tokenizer.encode(text, "all")
    # 保存分词器（TODO 应该使用一个适当的临时目录）
    tokenizer.save("test_tokenizer_tmp")
    # 重新加载分词器
    tokenizer = RegexTokenizer()
    tokenizer.load("test_tokenizer_tmp.model")
    # 验证 decode(encode(x)) == x
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(tokenizer.encode(text, "all")) == text
    assert tokenizer.encode(text, "all") == ids
    # 删除临时文件
    for file in ["test_tokenizer_tmp.model", "test_tokenizer_tmp.vocab"]:
        os.remove(file)

if __name__ == "__main__":
    pytest.main()
