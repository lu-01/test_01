[TOC]

# 数据处理（processors）
## \_\_init__.py
### def build_processor
#### 介绍
```py
def build_processor(args, tokenizer):
    """
    构建数据处理器。

    Args:
        args: 配置参数对象，包含数据集类型和模型类型等信息。
        tokenizer: 分词器对象，用于处理文本。

    Returns:
        processor: 数据处理器对象，用于加载和处理数据。
    """
```
#### 步骤
1. 检查数据集类型是否有效：args.dataset_type
2. 根据数据集类型设置训练、验证和测试文件路径
3. 如果模型类型为 "base"，加载最大跨度数量字典
4. 初始化多参数处理器：MultiargProcessor
5. 返回数据处理器对象
## processor_base.py
### class DSET_processor
#### 介绍
```py
class DSET_processor:
    """
    数据处理器类，用于加载和处理不同数据集（ACE、RAMS、WikiEvent）的事件数据。
    """
```
#### 步骤
1. 初始化数据处理器：\_\_init__
    1. 设置 args 和 token
    2. 读取角色模板和参数字典
    3. 设置 合并函数 collate_fn = None
2. 读取 JSONLines 格式的文件：\_read_jsonlines
    1. lines: 文件中的所有行（列表形式）
3. 读取 JSON 格式的文件：\_read_json
    1. 文件内容（字典形式）
4. 读取角色模板和参数字典：
    1. template_dict: 模板字典，键为事件类型和参数类型，值为模板字符
        - template_dict['ArtifactExistence.DamageDestroyDisableDismantle.Damage_Damager'] = 'to be continue'
    2. role_dict: 参数字典，键为事件类型，值为参数列表
        - role_dict['ArtifactExistence.DamageDestroyDisableDismantle.Damage'] = ['Damager', 'Artifact', 'Instrument', 'Place']

## processor_multiarg.py
### class ArgumentExtractionDataset(Dataset)
#### 介绍
```py
class ArgumentExtractionDataset(Dataset):
    """
    参数抽取数据集类。
    """
```
#### 步骤
1. 初始化特征列表features：\_\_init__
2. 返回数据集的大小：\_\_len__
3. 获取指定索引的特征：\_\_getitem__
4. 将多个样本合并为一个批次：collate_fn
    1. 待补充
### class MultiargProcessor(DSET_processor)
#### 介绍
```py
class MultiargProcessor(DSET_processor):
    """
    多参数处理器类，用于处理多参数事件抽取任务。
    """
```
#### 步骤
1. \_\_init__
    1. 设置解码器输入模式
    2. 设置合并函数：将多个样本合并为一个批次
2. 设置解码器输入模式：set_dec_input
    1. 参数查询：model_type == "base"
    2. 提示查询：model_type == "paie"
3. 
# utils.py
## set_seed
1. 设置随机种子以确保结果可复现
## 待补充