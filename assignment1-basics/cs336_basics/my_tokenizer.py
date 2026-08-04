from .my_bpe import train_bpe

import os
import regex as re

class Tokenizer:

    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            self.special_tokens=[]
        else:
            self.special_tokens=special_tokens

        self.special_tokens_bytes = [i.encode("utf-8") for i in self.special_tokens]

        self.bytes_to_id = {b: i for i, b in vocab.items()}

        self.merges_int = [(self.bytes_to_id[i[0]],self.bytes_to_id[i[1]]) for i in merges]

        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def encode(self,text:str):
        chunk_bytes_list=[]

        if self.special_tokens:
                # 构造特殊字符匹配正则
                special_pattern = re.compile(
                    "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
                )
        
                for part in re.split(special_pattern, text):
                    if not part:
                        continue
                    if part in self.special_tokens:
                        # 关键：特殊 token 作为一个整体放入，保护其不被正则切割，不参与后续 merge
                        chunk_bytes_list.append(part.encode("utf-8"))
                    else:
                        for chunk in re.findall(self.pattern, part):
                            chunk_bytes_list.append(list(chunk.encode("utf-8")))
        else:
            for chunk in re.findall(self.pattern, text):
                chunk_bytes_list.append(list(chunk.encode("utf-8")))
        

        # sorted_special_tokens = sorted(self.special_tokens, key=lambda s: len(s), reverse=True)
        # pattern = "(" +"|".join(re.escape(t) for t in sorted_special_tokens)+")"
        # parts = re.split(pattern,text)

        text_bytes = []

        for part in chunk_bytes_list:
            if part in self.special_tokens_bytes:
                text_bytes.append([self.bytes_to_id[part]])
            else:
                temp = [ self.bytes_to_id[bytes([i])] for i in part]
                text_bytes.append(temp)

        

        
        for ids in text_bytes:
            new_id = 256 + len(self.special_tokens)
            for merge in self.merges_int:
                i=0
                while i < len(ids):
                    if ids[i]==merge[0] and i < len(ids)-1 and ids[i+1] == merge[1]:
                        ids.pop(i)
                        ids.pop(i)
                        ids.insert(i,new_id)
                        i+=1
                    else:
                        i+=1
                        continue
                new_id+=1

        ans = []

        for ids in text_bytes:
            for i in ids:
                ans.append(i)

        return ans


    def encode_iterable(self, iterable):
        """给定一个字符串的可迭代对象（如 Python 文件句柄），
        返回一个惰性产出 token ID 的生成器。
        用于内存高效地 tokenize 无法直接加载到内存的大文件。"""
        for line in iterable:
            if not line:
                continue
            # 逐句/逐行编码并惰性产出
            yield from self.encode(line)
            

    def decode(self, ids: list[int]):
        ans_text = []
        for i in ids:
            ans_text.append(self.vocab[i])

        full_bytes = b''.join(ans_text)

        text = full_bytes.decode("utf-8", errors="replace")

        return text


    


