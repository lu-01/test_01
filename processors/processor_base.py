import csv
import json
import ipdb
import jsonlines
import torch

from random import sample
from itertools import chain
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import copy                             
import logging
logger = logging.getLogger(__name__)  # 设置日志记录器


class Event:
    """
    表示一个事件的类，包含事件的基本信息。
    """
    def __init__(self, doc_id, sent_id, sent, event_type, event_trigger, event_args, full_text, first_word_locs=None):
        """
        初始化事件对象。

        Args:
            doc_id: 文档 ID。
            sent_id: 句子 ID。
            sent: 句子内容（分词后的列表）。
            event_type: 事件类型。
            event_trigger: 事件触发器信息（字典）。
            event_args: 事件参数列表。
            full_text: 文档的完整文本。
            first_word_locs: 每个句子第一个单词的位置（可选）。
        """
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.sent = sent
        self.type = event_type
        self.trigger = event_trigger
        self.args = event_args
        self.full_text = full_text
        self.first_word_locs = first_word_locs

    def __str__(self):
        """
        返回事件对象的字符串表示。
        """
        return self.__repr__()

    def __repr__(self):
        """
        返回事件对象的详细字符串表示。
        """
        s = ""
        s += "doc id: {}\n".format(self.doc_id)
        s += "sent id: {}\n".format(self.sent_id)
        s += "text: {}\n".format(" ".join(self.sent))
        s += "event_type: {}\n".format(self.type)
        s += "trigger: {}\n".format(self.trigger['text'])
        for arg in self.args:
            s += "arg {}: {} ({}, {})\n".format(arg['role'], arg['text'], arg['start'], arg['end'])
        s += "----------------------------------------------\n"
        return s


class InputFeatures:
    """
    表示单个数据样本的特征类。
    """
    def __init__(self, example_id, feature_id, 
                 enc_text, dec_text,
                 enc_tokens, dec_tokens, 
                 old_tok_to_new_tok_index,  
                 event_type, event_trigger, argument_type,
                 enc_input_ids, enc_mask_ids, 
                 dec_input_ids, dec_mask_ids,
                 answer_text, start_position=None, end_position=None):
        """
        初始化特征对象。

        Args:
            example_id: 样本 ID。
            feature_id: 特征 ID。
            enc_text: 编码器输入文本。
            dec_text: 解码器输入文本。
            enc_tokens: 编码器分词后的 token 列表。
            dec_tokens: 解码器分词后的 token 列表。
            old_tok_to_new_tok_index: 旧到新 token 的索引映射。
            event_type: 事件类型。
            event_trigger: 事件触发器文本。
            argument_type: 参数类型。
            enc_input_ids: 编码器输入 ID。
            enc_mask_ids: 编码器掩码 ID。
            dec_input_ids: 解码器输入 ID。
            dec_mask_ids: 解码器掩码 ID。
            answer_text: 答案文本。
            start_position: 答案起始位置（可选）。
            end_position: 答案结束位置（可选）。
        """
        self.example_id = example_id
        self.feature_id = feature_id
        self.enc_text = enc_text
        self.dec_text = dec_text
        self.enc_tokens = enc_tokens
        self.dec_tokens = dec_tokens
        self.old_tok_to_new_tok_index = old_tok_to_new_tok_index
        self.event_type = event_type
        self.event_trigger = event_trigger
        self.argument_type = argument_type
        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids
        self.dec_input_ids = dec_input_ids
        self.dec_mask_ids = dec_mask_ids
        self.answer_text = answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        """
        返回特征对象的字符串表示。
        """
        return self.__repr__()

    def __repr__(self):
        """
        返回特征对象的详细字符串表示。
        """
        s = "" 
        s += "example_id: {}\n".format(self.example_id)
        s += "event_type: {}\n".format(self.event_type)
        s += "trigger_word: {}\n".format(self.event_trigger)
        s += "argument_type: {}\n".format(self.argument_type)
        s += "enc_tokens: {}\n".format(self.enc_tokens)
        s += "dec_tokens: {}\n".format(self.dec_tokens)
        s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
        s += "enc_input_ids: {}\n".format(self.enc_input_ids)
        s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
        s += "dec_input_ids: {}\n".format(self.dec_input_ids)
        s += "dec_mask_ids: {}\n".format(self.dec_mask_ids)
        s += "answer_text: {}\n".format(self.answer_text)
        s += "start_position: {}\n".format(self.start_position)
        s += "end_position: {}\n".format(self.end_position) 
        return s


class DSET_processor:
    """
    数据处理器类，用于加载和处理不同数据集（ACE、RAMS、WikiEvent）的事件数据。
    """
    def __init__(self, args, tokenizer):
        """
        初始化数据处理器。

        Args:
            args: 配置参数对象。
            tokenizer: 分词器对象，用于处理文本。
        """
        self.args = args
        self.tokenizer = tokenizer
        logger.info(f"正在从 {self.args.role_path} 加载角色模板和参数字典 (Loading role templates and argument dictionary from {self.args.role_path})")
        self.template_dict, self.argument_dict = self._read_roles(self.args.role_path)  # 读取角色模板和参数字典
        logger.info("角色模板和参数字典加载成功 (Role templates and argument dictionary loaded successfully).")
        self.collate_fn = None  # 数据加载器的合并函数（默认为 None）

    def _read_jsonlines(self, input_file):
        """
        读取 JSONLines 格式的文件。

        Args:
            input_file: 文件路径。

        Returns:
            lines: 文件中的所有行（列表形式）。
        """
        lines = []
        with jsonlines.open(input_file) as reader:
            for obj in reader:
                lines.append(obj)
        return lines

    def _read_json(self, input_file):
        """
        读取 JSON 格式的文件。

        Args:
            input_file: 文件路径。

        Returns:
            文件内容（字典形式）。
        """
        with open(input_file, "r", encoding='utf-8') as f:
            return json.load(f)

    def _read_roles(self, role_path):
        """
        读取角色模板和参数字典。

        Args:
            role_path: 角色模板文件路径。

        Returns:
            template_dict: 模板字典，键为事件类型和参数类型，值为模板字符串。
            role_dict: 参数字典，键为事件类型，值为参数列表。
        """
        template_dict = {}
        role_dict = {}

        with open(role_path, "r", encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                event_type_arg, template = line
                template_dict[event_type_arg] = template
                
                event_type, arg = event_type_arg.split('_')
                if event_type not in role_dict:
                    role_dict[event_type] = []
                role_dict[event_type].append(arg)

        return template_dict, role_dict

    def _create_example_ace(self, lines):
        """
        创建 ACE 数据集的事件样本。

        Args:
            lines: 数据集的所有行。

        Returns:
            examples: 事件样本列表。
        """
        examples = []
        for doc_idx, line in enumerate(lines):
            if not line['event']:
                continue
            events = line['event']
            offset = line['s_start']
            full_text = copy.deepcopy(line['sentence'])
            text = line['sentence']
            for event_idx, event in enumerate(events):
                event_type = event[0][1]
                event_trigger = dict()
                start = event[0][0] - offset
                end = start + 1
                event_trigger['start'] = start
                event_trigger['end'] = end
                event_trigger['text'] = " ".join(text[start:end])
                event_trigger['offset'] = offset

                event_args = list()
                for arg_info in event[1:]:
                    arg = dict()
                    start = arg_info[0] - offset
                    end = arg_info[1] - offset + 1
                    role = arg_info[2]
                    arg['start'] = start
                    arg['end'] = end
                    arg['role'] = role
                    arg['text'] = " ".join(text[start:end])
                    event_args.append(arg)

                examples.append(Event(doc_idx, event_idx, text, event_type, event_trigger, event_args, full_text))
            
        print("{} examples collected.".format(len(examples)))
        return examples

    def _create_example_rams(self, lines):
        """
        创建 RAMS 数据集的事件样本。

        Args:
            lines: 数据集的所有行。

        Returns:
            examples: 事件样本列表。
        """
        W = self.args.window_size  # 窗口大小
        assert(W % 2 == 0)
        all_args_num = 0

        examples = []
        for line in lines:
            if len(line["evt_triggers"]) == 0:
                continue
            doc_key = line["doc_key"]
            events = line["evt_triggers"]

            full_text = copy.deepcopy(list(chain(*line['sentences'])))
            cut_text = list(chain(*line['sentences']))
            sent_length = sum([len(sent) for sent in line['sentences']])

            text_tmp = []
            first_word_locs = []
            for sent in line["sentences"]:
                first_word_locs.append(len(text_tmp))
                text_tmp += sent

            for event_idx, event in enumerate(events):                
                event_trigger = dict()
                event_trigger['start'] = event[0]
                event_trigger['end'] = event[1] + 1
                event_trigger['text'] = " ".join(full_text[event_trigger['start']:event_trigger['end']])
                event_type = event[2][0][0]

                offset, min_s, max_e = 0, 0, W + 1
                event_trigger['offset'] = offset
                if sent_length > W + 1:
                    if event_trigger['end'] <= W // 2:  # 触发词位于句子前部
                        cut_text = full_text[:(W + 1)]
                    else:  # 触发词位于句子后部
                        offset = sent_length - (W + 1)
                        min_s += offset
                        max_e += offset
                        event_trigger['start'] -= offset
                        event_trigger['end'] -= offset 
                        event_trigger['offset'] = offset
                        cut_text = full_text[-(W + 1):]

                event_args = list()
                for arg_info in line["gold_evt_links"]:
                    if arg_info[0][0] == event[0] and arg_info[0][1] == event[1]:  # 匹配触发词跨度    
                        all_args_num += 1

                        evt_arg = dict()
                        evt_arg['start'] = arg_info[1][0]
                        evt_arg['end'] = arg_info[1][1] + 1
                        evt_arg['text'] = " ".join(full_text[evt_arg['start']:evt_arg['end']])
                        evt_arg['role'] = arg_info[2].split('arg', maxsplit=1)[-1][2:]
                        if evt_arg['start'] < min_s or evt_arg['end'] > max_e:
                            self.invalid_arg_num += 1
                        else:
                            evt_arg['start'] -= offset
                            evt_arg['end'] -= offset 
                            event_args.append(evt_arg)

                if event_idx > 0:
                    examples.append(Event(doc_key + str(event_idx), None, cut_text, event_type, event_trigger, event_args, full_text, first_word_locs))
                else:
                    examples.append(Event(doc_key, None, cut_text, event_type, event_trigger, event_args, full_text, first_word_locs))
            
        print("{} examples collected. {} arguments dropped.".format(len(examples), self.invalid_arg_num))
        return examples

    def _create_example_wikievent(self, lines):
        """
        创建 WikiEvent 数据集的事件样本。

        Args:
            lines: 数据集的所有行。

        Returns:
            examples: 事件样本列表。
        """
        W = self.args.window_size
        assert(W % 2 == 0)
        all_args_num = 0

        examples = []
        for line in lines:
            entity_dict = {entity['id']: entity for entity in line['entity_mentions']}
            events = line["event_mentions"]
            if not events:
                continue
            doc_key = line["doc_id"]
            full_text = line['tokens']
            sent_length = len(full_text)

            curr_loc = 0
            first_word_locs = []
            for sent in line["sentences"]:
                first_word_locs.append(curr_loc)
                curr_loc += len(sent[0])

            for event in events:
                event_type = event['event_type']
                cut_text = full_text
                event_trigger = event['trigger']

                offset, min_s, max_e = 0, 0, W + 1
                if sent_length > W + 1:
                    if event_trigger['end'] <= W // 2:  # 触发词位于句子前部
                        cut_text = full_text[:(W + 1)]
                    elif event_trigger['start'] >= sent_length - W / 2:  # 触发词位于句子后部
                        offset = sent_length - (W + 1)
                        min_s += offset
                        max_e += offset
                        event_trigger['start'] -= offset
                        event_trigger['end'] -= offset 
                        cut_text = full_text[-(W + 1):]
                    else:
                        offset = event_trigger['start'] - W // 2
                        min_s += offset
                        max_e += offset
                        event_trigger['start'] -= offset
                        event_trigger['end'] -= offset 
                        cut_text = full_text[offset:(offset + W + 1)]
                event_trigger['offset'] = offset
                        
                event_args = list()
                for arg_info in event['arguments']:
                    all_args_num += 1

                    evt_arg = dict()
                    arg_entity = entity_dict[arg_info['entity_id']]
                    evt_arg['start'] = arg_entity['start']
                    evt_arg['end'] = arg_entity['end']
                    evt_arg['text'] = arg_info['text']
                    evt_arg['role'] = arg_info['role']
                    if evt_arg['start'] < min_s or evt_arg['end'] > max_e:
                        self.invalid_arg_num += 1
                    else:
                        evt_arg['start'] -= offset
                        evt_arg['end'] -= offset 
                        event_args.append(evt_arg)
                examples.append(Event(doc_key, None, cut_text, event_type, event_trigger, event_args, full_text, first_word_locs))

        logger.info("{} examples collected. {} dropped.".format(len(examples), self.invalid_arg_num))
        return examples

    def create_example(self, file_path):
        """
        根据数据集类型创建事件样本。

        Args:
            file_path: 数据文件路径。

        Returns:
            examples: 事件样本列表。
        """
        self.invalid_arg_num = 0
        if self.args.dataset_type == 'ace_eeqa':
            lines = self._read_jsonlines(file_path)
            return self._create_example_ace(lines)
        elif self.args.dataset_type == 'rams':
            lines = self._read_jsonlines(file_path)
            return self._create_example_rams(lines)
        elif self.args.dataset_type == 'wikievent':
            lines = self._read_jsonlines(file_path)
            return self._create_example_wikievent(lines)
        else:
            raise NotImplementedError()

    def convert_examples_to_features(self, examples):
        """
        将事件样本转换为特征。

        Args:
            examples: 事件样本列表。

        Returns:
            features: 特征列表。
        """
        features = []
        for (example_idx, example) in enumerate(examples):
            sent = example.sent  
            event_type = example.type
            event_args = example.args
            event_trigger = example.trigger['text']
            event_args_name = [arg['role'] for arg in event_args]
            enc_text = " ".join(sent)

            old_tok_to_char_index = []  # 旧 token：基于原始分词
            old_tok_to_new_tok_index = []  # 新 token：基于 BART 分词

            curr = 0
            for tok in sent:
                old_tok_to_char_index.append(curr)
                curr += len(tok) + 1
            assert(len(old_tok_to_char_index) == len(sent))

            enc = self.tokenizer(enc_text)
            enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
            enc_tokens = self.tokenizer.convert_ids_to_tokens(enc_input_ids)  
            while len(enc_input_ids) < self.args.max_enc_seq_length:
                enc_input_ids.append(self.tokenizer.pad_token_id)
                enc_mask_ids.append(self.args.pad_mask_token)
            
            for char_idx in old_tok_to_char_index:
                new_tok = enc.char_to_token(char_idx)
                old_tok_to_new_tok_index.append(new_tok)    
    
            for arg in self.argument_dict[event_type.replace(':', '.')]:
                dec_text = 'Argument ' + arg + ' in ' + event_trigger + ' event ?' + " "
                     
                dec = self.tokenizer(dec_text)
                dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]
                dec_tokens = self.tokenizer.convert_ids_to_tokens(dec_input_ids) 
                while len(dec_input_ids) < self.args.max_dec_seq_length:
                    dec_input_ids.append(self.tokenizer.pad_token_id)
                    dec_mask_ids.append(self.args.pad_mask_token)
        
                start_position, end_position, answer_text = None, None, None
                if arg in event_args_name:
                    arg_idx = event_args_name.index(arg)
                    event_arg_info = event_args[arg_idx]
                    answer_text = event_arg_info['text']
                    start_old, end_old = event_arg_info['start'], event_arg_info['end']
                    start_position = old_tok_to_new_tok_index[start_old]
                    end_position = old_tok_to_new_tok_index[end_old] if end_old < len(old_tok_to_new_tok_index) else old_tok_to_new_tok_index[-1] + 1 
                else:
                    start_position, end_position = 0, 0
                    answer_text = "__ No answer __"

                feature_idx = len(features)
                features.append(
                      InputFeatures(example_idx, feature_idx, 
                                    enc_text, dec_text,
                                    enc_tokens, dec_tokens,
                                    old_tok_to_new_tok_index,
                                    event_type, event_trigger, arg,
                                    enc_input_ids, enc_mask_ids, 
                                    dec_input_ids, dec_mask_ids,
                                    answer_text, start_position, end_position
                                )
                )
        return features

    def convert_features_to_dataset(self, features):
        """
        将特征转换为数据集。

        Args:
            features: 特征列表。

        Returns:
            dataset: 数据集对象。
        """
        all_enc_input_ids = torch.tensor([f.enc_input_ids for f in features], dtype=torch.long).to(self.args.device)
        all_enc_mask_ids = torch.tensor([f.enc_mask_ids for f in features], dtype=torch.long).to(self.args.device)
        all_dec_input_ids = torch.tensor([f.dec_input_ids for f in features], dtype=torch.long).to(self.args.device)
        all_dec_mask_ids = torch.tensor([f.dec_mask_ids for f in features], dtype=torch.long).to(self.args.device)
        
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long).to(self.args.device)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long).to(self.args.device)
        all_example_idx = torch.tensor([f.example_id for f in features], dtype=torch.long).to(self.args.device)
        all_feature_idx = torch.tensor([f.feature_id for f in features], dtype=torch.long).to(self.args.device)

        dataset = TensorDataset(all_enc_input_ids, all_enc_mask_ids,
                                all_dec_input_ids, all_dec_mask_ids,
                                all_start_positions, all_end_positions,
                                all_example_idx, all_feature_idx)
        return dataset

    def generate_dataloader(self, set_type):
        """
        生成数据加载器。

        Args:
            set_type: 数据集类型（'train'、'dev' 或 'test'）。

        Returns:
            examples: 事件样本列表。
            features: 特征列表。
            dataloader: 数据加载器。
            invalid_arg_num: 无效参数数量。
        """
        assert (set_type in ['train', 'dev', 'test'])
        if set_type == 'train':
            file_path = self.args.train_file
        elif set_type == 'dev':
            file_path = self.args.dev_file
        else:
            file_path = self.args.test_file
        
        examples = self.create_example(file_path)
        if set_type == 'train' and self.args.keep_ratio < 1.0:
            sample_num = int(len(examples) * self.args.keep_ratio)
            examples = sample(examples, sample_num)
            logger.info("Few shot setting: keep ratio {}. Only {} training samples remained.".format(
                self.args.keep_ratio, len(examples))
            )

        features = self.convert_examples_to_features(examples)
        dataset = self.convert_features_to_dataset(features)

        if set_type != 'train':
            dataset_sampler = SequentialSampler(dataset)
        else:
            dataset_sampler = RandomSampler(dataset)
        if self.collate_fn:
            dataloader = DataLoader(dataset, sampler=dataset_sampler, batch_size=self.args.batch_size, collate_fn=self.collate_fn)
        else:
            dataloader = DataLoader(dataset, sampler=dataset_sampler, batch_size=self.args.batch_size)
        return examples, features, dataloader, self.invalid_arg_num