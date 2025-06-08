import os
import re
import sys
sys.path.append("../")  # 添加上级目录到系统路径，方便模块导入
import torch
import numpy as np

from torch.utils.data import Dataset
from processors.processor_base import DSET_processor  # 导入基础数据处理器
from utils import EXTERNAL_TOKENS, _PREDEFINED_QUERY_TEMPLATE  # 导入特殊标记和预定义查询模板
import logging
logger = logging.getLogger(__name__)  # 设置日志记录器

class InputFeatures(object):
    """
    表示单个数据样本的特征类。
    """
    def __init__(self, example_id, feature_id, 
                event_type, event_trigger,
                enc_text, enc_input_ids, enc_mask_ids, 
                dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
                arg_quries, arg_joint_prompt, target_info,
                old_tok_to_new_tok_index=None, full_text=None, arg_list=None):
        """
        初始化特征对象。

        Args:
            example_id: 样本文档名。
            feature_id: 特征 ID。
            event_type: 事件类型。
            event_trigger: 事件触发器信息。
            enc_text: 编码器输入文本。
            enc_input_ids: 编码器 输入。
            enc_mask_ids: 编码器 掩码。
            dec_prompt_text: 解码器  提示文本。
            dec_prompt_ids: 解码器 提示 ID。
            dec_prompt_mask_ids: 解码器 提示掩码 ID。
            arg_quries: 参数查询信息。
            arg_joint_prompt: 参数联合提示信息。（提示模板中角色信息）
            target_info: 目标信息。（实际事件的角色信息）
            old_tok_to_new_tok_index: 旧到新 token 的索引映射。
            full_text: 事件文本的完整内容。
            arg_list: 参数列表。（当前事件类型的所有参数角色）
        """
        self.example_id = example_id
        self.feature_id = feature_id
        self.event_type = event_type
        self.event_trigger = event_trigger
        
        self.enc_text = enc_text
        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids

        self.dec_prompt_texts = dec_prompt_text
        self.dec_prompt_ids = dec_prompt_ids
        self.dec_prompt_mask_ids = dec_prompt_mask_ids

        if arg_quries is not None:      # 使用参数查询时
            self.dec_arg_query_ids = [v[0] for k, v in arg_quries.items()]
            self.dec_arg_query_masks = [v[1] for k, v in arg_quries.items()]
            self.dec_arg_start_positions = [v[2] for k, v in arg_quries.items()]
            self.dec_arg_end_positions = [v[3] for k, v in arg_quries.items()]
            self.start_position_ids = [v['span_s'] for k, v in target_info.items()]
            self.end_position_ids = [v['span_e'] for k, v in target_info.items()]
        else:           # 使用提示查询时
            self.dec_arg_query_ids = None
            self.dec_arg_query_masks = None
        
        self.arg_joint_prompt = arg_joint_prompt
        self.target_info = target_info
        self.old_tok_to_new_tok_index = old_tok_to_new_tok_index
        self.full_text = full_text
        self.arg_list = arg_list

    def init_pred(self):
        """
        初始化预测字典。
        """
        self.pred_dict_tok = dict()
        self.pred_dict_word = dict()

    def add_pred(self, role, span, dset_type):
        """
        添加预测结果。

        Args:
            role: 参数角色。
            span: 预测的跨度。
            dset_type: 数据集类型。
        """
        if role not in self.pred_dict_tok:
            self.pred_dict_tok[role] = list()
        if span not in self.pred_dict_tok[role]:
            self.pred_dict_tok[role].append(span)

            if span != (0, 0):
                if role not in self.pred_dict_word:
                    self.pred_dict_word[role] = list()
                word_span = self.get_word_span(span, dset_type)  # 将 token 跨度转换为 word 跨度
                if word_span not in self.pred_dict_word[role]:
                    self.pred_dict_word[role].append(word_span)

    def set_gt(self, model_type, dset_type):
        """
        设置 ground truth（真实值）。

        Args:
            model_type: 模型类型。
            dset_type: 数据集类型。
        """
        self.gt_dict_tok = dict()
        if model_type == 'base':
            for k, v in self.target_info.items():
                span_s = list(np.where(v["span_s"])[0])
                span_e = list(np.where(v["span_e"])[0])
                self.gt_dict_tok[k] = [(s, e) for (s, e) in zip(span_s, span_e)]
        elif "paie" in model_type:
            for k, v in self.target_info.items():
                self.gt_dict_tok[k] = [(s, e) for (s, e) in zip(v["span_s"], v["span_e"])]
        else:
            assert False, "Unsupported model type"

        self.gt_dict_word = dict()
        for role in self.gt_dict_tok:
            for span in self.gt_dict_tok[role]:
                if span != (0, 0):
                    if role not in self.gt_dict_word:
                        self.gt_dict_word[role] = list()
                    word_span = self.get_word_span(span, dset_type)
                    self.gt_dict_word[role].append(word_span)

    @property
    def old_tok_index(self):
        """
        获取新 token 到旧 token 的索引映射。
        """
        new_tok_index_to_old_tok_index = dict()
        for old_tok_id, (new_tok_id_s, new_tok_id_e) in enumerate(self.old_tok_to_new_tok_index):
            for j in range(new_tok_id_s, new_tok_id_e):
                new_tok_index_to_old_tok_index[j] = old_tok_id 
        return new_tok_index_to_old_tok_index

    def get_word_span(self, span, dset_type):
        """
        将 token 跨度转换为 word 跨度。

        Args:
            span: token 跨度。
            dset_type: 数据集类型。

        Returns:
            word_span: 转换后的 word 跨度。
        """
        if span == (0, 0):
            raise AssertionError("Invalid span")
        offset = 0 if dset_type == 'ace_eeqa' else self.event_trigger[2]
        span = list(span)
        span[0] = min(span[0], max(self.old_tok_index.keys()))
        span[1] = max(span[1] - 1, min(self.old_tok_index.keys()))

        while span[0] not in self.old_tok_index:
            span[0] += 1 
        span_s = self.old_tok_index[span[0]] + offset
        while span[1] not in self.old_tok_index:
            span[1] -= 1 
        span_e = self.old_tok_index[span[1]] + offset
        while span_e < span_s:
            span_e += 1
        return (span_s, span_e)

    def __repr__(self):
        """
        返回特征对象的字符串表示。
        """
        s = "" 
        s += "example_id: {}\n".format(self.example_id)
        s += "event_type: {}\n".format(self.event_type)
        s += "trigger_word: {}\n".format(self.event_trigger)
        s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
        s += "enc_input_ids: {}\n".format(self.enc_input_ids)
        s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
        s += "dec_prompt_ids: {}\n".format(self.dec_prompt_ids)
        s += "dec_prompt_mask_ids: {}\n".format(self.dec_prompt_mask_ids)
        return s


class ArgumentExtractionDataset(Dataset):
    """
    参数抽取数据集类。
    """
    def __init__(self, features):
        """
        初始化数据集。

        Args:
            features: 特征列表。
        """
        self.features = features
    
    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        获取指定索引的特征。

        Args:
            idx: 索引。

        Returns:
            特征对象。
        """
        return self.features[idx]
 
    @staticmethod
    def collate_fn(batch):
        """
        合并函数，用于将多个样本合并为一个批次。

        Args:
            batch: 样本列表。

        Returns:
            合并后的批次数据。
        """
        enc_input_ids = torch.tensor([f.enc_input_ids for f in batch])
        enc_mask_ids = torch.tensor([f.enc_mask_ids for f in batch])

        if batch[0].dec_prompt_ids is not None:
            dec_prompt_ids = torch.tensor([f.dec_prompt_ids for f in batch])
            dec_prompt_mask_ids = torch.tensor([f.dec_prompt_mask_ids for f in batch])
        else:
            dec_prompt_ids = None
            dec_prompt_mask_ids = None

        example_idx = [f.example_id for f in batch]
        feature_idx = torch.tensor([f.feature_id for f in batch])

        if batch[0].dec_arg_query_ids is not None:
            dec_arg_query_ids = [torch.LongTensor(f.dec_arg_query_ids) for f in batch]
            dec_arg_query_mask_ids = [torch.LongTensor(f.dec_arg_query_masks) for f in batch]
            dec_arg_start_positions = [torch.LongTensor(f.dec_arg_start_positions) for f in batch]
            dec_arg_end_positions = [torch.LongTensor(f.dec_arg_end_positions) for f in batch]
            start_position_ids = [torch.FloatTensor(f.start_position_ids) for f in batch]
            end_position_ids = [torch.FloatTensor(f.end_position_ids) for f in batch]
        else:
            dec_arg_query_ids = None
            dec_arg_query_mask_ids = None
            dec_arg_start_positions = None
            dec_arg_end_positions = None
            start_position_ids = None
            end_position_ids = None

        target_info = [f.target_info for f in batch]
        old_tok_to_new_tok_index = [f.old_tok_to_new_tok_index for f in batch]
        arg_joint_prompt = [f.arg_joint_prompt for f in batch]
        arg_lists = [f.arg_list for f in batch]

        return enc_input_ids, enc_mask_ids, \
               dec_arg_query_ids, dec_arg_query_mask_ids, \
               dec_prompt_ids, dec_prompt_mask_ids, \
               target_info, old_tok_to_new_tok_index, arg_joint_prompt, arg_lists, \
               example_idx, feature_idx, \
               dec_arg_start_positions, dec_arg_end_positions, \
               start_position_ids, end_position_ids


class MultiargProcessor(DSET_processor):
    """
    多参数处理器类，用于处理多参数事件抽取任务。
    """
    def __init__(self, args, tokenizer):
        """
        初始化处理器。

        Args:
            args: 配置参数对象。
            tokenizer: 分词器对象，用于处理文本。
        """
        logger.info(f"Entering class: {self.__class__.__name__}, function: {sys._getframe().f_code.co_name}")  # 类初始化开始日志
        super().__init__(args, tokenizer) 
        self.set_dec_input()  # 设置解码器输入模式
        self.collate_fn = ArgumentExtractionDataset.collate_fn  # 设置合并函数

        logger.info("设置合并函数和解码器输入模式完成")
        logger.info("Initialized MultiargProcessor with model_type: %s", args.model_type)
        logger.info(f"class: {self.__class__.__name__}, function: {sys._getframe().f_code.co_name} successfully")  # 类初始化结束日志


    def set_dec_input(self):
        """
        设置解码器输入模式（参数查询或提示查询）。
        """
        self.arg_query = False
        self.prompt_query = False
        if self.args.model_type == "base":
            self.arg_query = True
        elif "paie" in self.args.model_type:
            self.prompt_query = True
        else:
            raise NotImplementedError(f"Unexpected setting {self.args.model_type}")
        logger.info("Set decoder input mode: arg_query=%s, prompt_query=%s", self.arg_query, self.prompt_query)
     

    @staticmethod
    def _read_prompt_group(prompt_path):
        """
        读取提示组文件。

        Args:
            prompt_path: 提示组文件路径。

        Returns:
            prompts: 包含事件类型和对应提示的字典。
        """
        logger.info(f"Reading prompt group from {prompt_path}")
        with open(prompt_path) as f:
            lines = f.readlines()
        prompts = dict()
        for line in lines:
            if not line:
                continue
            event_type, prompt = line.split(":")
            prompts[event_type] = prompt
        logger.info("prompts示例:  key={}, value={}".format(event_type, prompt))
        logger.info(f"Loaded {len(prompts)} prompts from {prompt_path}")
        return prompts


    def create_dec_qury(self, arg, event_trigger):
        """
        创建解码器查询。

        Args:
            arg: 参数名称。
            event_trigger: 事件触发器文本。

        Returns:
            dec_input_ids: 解码器输入 ID。
            dec_mask_ids: 解码器掩码 ID。
            tok_prompt_s: 参数在解码器中的起始 token 索引。
            tok_prompt_e: 参数在解码器中的结束 token 索引。
        """
        dec_text = _PREDEFINED_QUERY_TEMPLATE.format(arg=arg, trigger=event_trigger)
                
        dec = self.tokenizer(dec_text)
        dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]

        # 填充到最大解码器序列长度
        while len(dec_input_ids) < self.args.max_dec_seq_length:
            dec_input_ids.append(self.tokenizer.pad_token_id)
            dec_mask_ids.append(self.args.pad_mask_token)

        # 查找参数在解码器文本中的位置
        matching_result = re.search(arg, dec_text)
        char_idx_s, char_idx_e = matching_result.span()
        char_idx_e -= 1
        tok_prompt_s = dec.char_to_token(char_idx_s)
        tok_prompt_e = dec.char_to_token(char_idx_e) + 1

        return dec_input_ids, dec_mask_ids, tok_prompt_s, tok_prompt_e


    def convert_examples_to_features(self, examples):
        """
        将事件样本转换为特征。
            文本编码（使用BART等预训练分词器）
            触发词和参数的位置标记
            提示模板（Prompt）处理（若启用）
        Args:
            examples: 事件样本列表。

        Returns:
            features: 特征列表。
        """
        logger.info(f"Entering class: {self.__class__.__name__}, function: {sys._getframe().f_code.co_name}")  # 函数开始日志
        # 提示查询时读取提示模板   key= event_type, value= prompt
        if self.prompt_query:
            prompts = self._read_prompt_group(self.args.prompt_path)

        if os.environ.get("DEBUG", False): 
            counter = [0, 0, 0]  # 用于调试的计数器
        features = []
        for example_idx, example in enumerate(examples):
            example_id = example.doc_id         # 文档id：scenario_en_kairos_14 
            sent = example.sent                 # 事件句子内容（分词后的列表）： ['the', '14th', 'century', ...]
            event_type = example.type           # 事件类型：Cognitive.IdentifyCategorize.Unspecified
            event_args = example.args           # 事件参数列表： [{'role': 'Cognitive.IdentifyCategorize.Unspecified', 'text': 'the 14th century', 'start': 0, 'end': 16, 'offset': 0}, ...]
     
            trigger_start, trigger_end = example.trigger['start'], example.trigger['end']
            # 扩展触发器的完整信息 ['discovered', [166, 167], 0]    
            event_trigger = [example.trigger['text'], [trigger_start, trigger_end], example.trigger['offset']]  

            event_args_name = [arg['role'] for arg in event_args]       # 提取参数角色名称列表
            if os.environ.get("DEBUG", False): 
                counter[2] += len(event_args_name)
            # 处理事件, 将触发器标记为 <t> 和 </t>
            sent = sent[:trigger_start] + ['<t>'] + sent[trigger_start:trigger_end] + ['</t>'] + sent[trigger_end:]
            enc_text = " ".join(sent)

            # 创建旧 token 到字符索引和新 token 的映射
            old_tok_to_char_index = []     # 旧 token：基于原始分词
            old_tok_to_new_tok_index = []  # 新 token：基于 BART 分词
            
            curr = 0
            for tok in sent:
                if tok not in EXTERNAL_TOKENS:      # 特殊标记，用于模型输入 ['<t>', '</t>']
                    old_tok_to_char_index.append([curr, curr + len(tok) - 1])  # 精确的字符起始和结束索引
                curr += len(tok) + 1
            # 调用分词器对文本进行编码，得到 input_ids（token id 序列）和 attention_mask（注意力掩码）
            enc = self.tokenizer(enc_text)
            enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
            # 编码后的序列长度超过了模型允许的最大长度 max_enc_seq_length，则抛出异常，提示需要增大最大长度。
            # 否则，如果长度不足，则通过补齐（append）
            if len(enc_input_ids) > self.args.max_enc_seq_length:
                raise ValueError(f"Please increase max_enc_seq_length above {len(enc_input_ids)}")
            while len(enc_input_ids) < self.args.max_enc_seq_length:
                enc_input_ids.append(self.tokenizer.pad_token_id)
                enc_mask_ids.append(self.args.pad_mask_token)
            # 处理旧 token 到新 token 的索引映射, 字符级索引[]转换为新 token 索引
            # BART 分词器将字符索引转换为 token 索引，与原始token索引不同
            for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(old_tok_to_char_index):
                new_tok_s = enc.char_to_token(char_idx_s)
                new_tok_e = enc.char_to_token(char_idx_e) + 1
                new_tok = [new_tok_s, new_tok_e]
                old_tok_to_new_tok_index.append(new_tok)    

            # 处理提示模板
            if self.prompt_query:
                dec_prompt_text = prompts[event_type].strip()      # 获取对应事件类型的提示模板
                if dec_prompt_text:
                    dec_prompt = self.tokenizer(dec_prompt_text)    # 对提示模板进行编码
                    dec_prompt_ids, dec_prompt_mask_ids = dec_prompt["input_ids"], dec_prompt["attention_mask"]
                    assert len(dec_prompt_ids) <= self.args.max_prompt_seq_length, f"\n{example}\n{arg_list}\n{dec_prompt_text}"
                    while len(dec_prompt_ids) < self.args.max_prompt_seq_length:    # 如果提示模板长度不足，则通过补齐
                        dec_prompt_ids.append(self.tokenizer.pad_token_id)
                        dec_prompt_mask_ids.append(self.args.pad_mask_token)
                else:
                    raise ValueError(f"no prompt provided for event: {event_type}")
            else:
                dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids = None, None, None
                
            arg_list = self.argument_dict[event_type.replace(':', '.')] 
            arg_quries = dict()
            arg_joint_prompt = dict()      # 各个参数角色在提示中的位置信息
            target_info = dict()           # 各个参数角色在当前事件中的位置信息
            if os.environ.get("DEBUG", False): 
                arg_set = set()
            for arg in arg_list:        # 遍历每个参数角色
                arg_query = None
                prompt_slots = None
                arg_target = {          # 参数角色在当前事件中的位置信息
                    "text": list(),
                    "span_s": list(),
                    "span_e": list()
                }

                if self.arg_query:
                    arg_query = self.create_dec_qury(arg, event_trigger[0])
                if self.prompt_query:
                    prompt_slots = {
                        "tok_s": list(),        # 5
                        "tok_e": list(),        # 7
                    }
                    
                    # 使用正则表达式查找参数在提示中的位置
                    for matching_result in re.finditer(r'\b' + re.escape(arg) + r'\b', dec_prompt_text.split('.')[0]): 
                        char_idx_s, char_idx_e = matching_result.span()     # 14， 24
                        char_idx_e -= 1
                        tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
                        tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
                        prompt_slots["tok_s"].append(tok_prompt_s)
                        prompt_slots["tok_e"].append(tok_prompt_e)

                answer_texts, start_positions, end_positions = list(), list(), list()
                if arg in event_args_name:      # 若参数存在于当前事件中
                    # 处理多次出现的参数
                    if os.environ.get("DEBUG", False): 
                        arg_set.add(arg)
                    arg_idxs = [i for i, x in enumerate(event_args_name) if x == arg]   # 相同参数出现的不同位置 
                    if os.environ.get("DEBUG", False): 
                        counter[0] += 1
                        counter[1] += len(arg_idxs)
                    # 多个参数角色可能对应多个位置
                    for arg_idx in arg_idxs:
                        event_arg_info = event_args[arg_idx]
                        answer_text = event_arg_info['text']
                        answer_texts.append(answer_text)
                        start_old, end_old = event_arg_info['start'], event_arg_info['end']
                        start_position = old_tok_to_new_tok_index[start_old][0]
                        start_positions.append(start_position)
                        end_position = old_tok_to_new_tok_index[end_old - 1][1]
                        end_positions.append(end_position)

                if self.arg_query:
                    arg_target["span_s"] = [1 if i in start_positions else 0 for i in range(self.args.max_enc_seq_length)]
                    arg_target["span_e"] = [1 if i in end_positions else 0 for i in range(self.args.max_enc_seq_length)]
                    if sum(arg_target["span_s"]) == 0:
                        arg_target["span_s"][0] = 1
                        arg_target["span_e"][0] = 1
                if self.prompt_query:
                    arg_target["span_s"] = start_positions
                    arg_target["span_e"] = end_positions

                arg_target["text"] = answer_texts
                arg_quries[arg] = arg_query
                arg_joint_prompt[arg] = prompt_slots
                target_info[arg] = arg_target

            if not self.arg_query:
                arg_quries = None       
            if not self.prompt_query:   
                arg_joint_prompt = None

            # 创建特征对象
            feature_idx = len(features)
            features.append(
                InputFeatures(
                    example_id, feature_idx, 
                    event_type, event_trigger,
                    enc_text, enc_input_ids, enc_mask_ids, 
                    dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
                    arg_quries, arg_joint_prompt, target_info,
                    old_tok_to_new_tok_index=old_tok_to_new_tok_index, 
                    full_text=example.full_text, 
                    arg_list=arg_list
                )
            )

        if os.environ.get("DEBUG", False): 
            print('\033[91m' + f"distinct/tot arg_role: {counter[0]}/{counter[1]} ({counter[2]})" + '\033[0m')
        logger.info(f"Converted {len(examples)} examples to {len(features)} features.")
        logger.info("第一个特征样本\n" + str(features[0]))  # 打印第一个特征样本的详细信息
        logger.info(f"class: {self.__class__.__name__}, function: {sys._getframe().f_code.co_name} successfully")  # 类初始化结束日志
        return features

    
    def convert_features_to_dataset(self, features):
        """
        将特征转换为数据集。

        Args:
            features: 特征列表。

        Returns:
            dataset: 参数抽取数据集对象。
        """
        logger.info(f"Entering class: {self.__class__.__name__}, function: {sys._getframe().f_code.co_name}")
        # 转换成dataset对象，无实际变化
        dataset = ArgumentExtractionDataset(features)
        logger.info(f"Converted {len(features)} features to dataset with shape: {len(dataset)}")  # 打印数据集大小
        logger.info(f"第一个数据集样本\n{dataset[0]}")  # 打印第一个数据集样本的详细信息
        logger.info(f"class: {self.__class__.__name__}, function: {sys._getframe().f_code.co_name} successfully")  # 类初始化结束日志
        return dataset