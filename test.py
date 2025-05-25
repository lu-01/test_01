print("test")
import spacy
from typing_extensions import Self
nlp = spacy.load("en_core_web_sm")  # 单独测试是否报错

import torch
print(torch.__version__)          # 查看版本
print(torch.cuda.is_available())  # 应返回 False

