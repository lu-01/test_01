import os
import re
import time
import string
import random
import logging
logger = logging.getLogger(__name__)  # 设置日志记录器
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment  # 用于匈牙利算法的线性分配

# 特殊标记，用于模型输入
EXTERNAL_TOKENS = ['<t>', '</t>']
# 预定义的查询模板
_PREDEFINED_QUERY_TEMPLATE = "Argument: {arg:}. Trigger: {trigger:} "

def set_seed(args):
    """
    设置随机种子以确保结果可复现。
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def count_time(f):
    """
    装饰器，用于统计函数执行时间。
    """
    def run(**kw):
        time1 = time.time()
        result = f(**kw)
        time2 = time.time()
        logger.info("The time of executing {}: {}".format(f.__name__, time2-time1))
        return result
    return run


def hungarian_matcher(predicted_spans, target_spans):
    """
    使用匈牙利算法匹配预测跨度和目标跨度。

    Args:
        predicted_spans: 预测的跨度列表，每个跨度为 [start, end]
        target_spans: 目标的跨度列表，每个跨度为 [start, end]

    Returns:
        indices: (index_i, index_j)，分别表示预测和目标的匹配索引
    """
    # 计算跨度之间的 L1 距离
    cost_spans = torch.cdist(torch.FloatTensor(predicted_spans).unsqueeze(0), torch.FloatTensor(target_spans).unsqueeze(0), p=1)
    indices = linear_sum_assignment(cost_spans.squeeze(0)) 
    return [torch.as_tensor(indices[0], dtype=torch.int64), torch.as_tensor(indices[1], dtype=torch.int64)]


def _normalize_answer(s):
    """
    标准化答案文本（类似于 SQuAD 数据集的处理方式）。
    包括：小写、去除标点符号、去除冠词、修正多余空格。

    Args:
        s: 输入字符串

    Returns:
        标准化后的字符串
    """
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    s_normalized = white_space_fix(remove_articles(remove_punc(lower(s))))
    return s_normalized


def get_best_span(start_logit, end_logit, old_tok_to_new_tok_index, max_span_length):
    """
    获取最佳跨度（基于分数）。

    Args:
        start_logit: 起始位置的 logits
        end_logit: 结束位置的 logits
        old_tok_to_new_tok_index: 旧到新 token 的索引映射
        max_span_length: 最大跨度长度

    Returns:
        best_answer_span: 最佳跨度 (start_index, end_index)
    """
    best_score = start_logit[0] + end_logit[0]
    best_answer_span = (0, 0)
    context_length = len(old_tok_to_new_tok_index)

    for start in range(context_length):
        for end in range(start+1, min(context_length, start+max_span_length+1)):
            start_index = old_tok_to_new_tok_index[start][0]  # 起始 token 索引
            end_index = old_tok_to_new_tok_index[end-1][1]  # 结束 token 索引

            score = start_logit[start_index] + end_logit[end_index]
            answer_span = (start_index, end_index)
            if score > best_score:
                best_score = score
                best_answer_span = answer_span

    return best_answer_span


def get_best_span_simple(start_logit, end_logit):
    """
    简化版本的最佳跨度获取（无复杂约束）。

    Args:
        start_logit: 起始位置的 logits
        end_logit: 结束位置的 logits

    Returns:
        [s_idx, e_idx]: 最佳跨度的起始和结束索引
    """
    _, s_idx = torch.max(start_logit, dim=0)
    _, e_idx = torch.max(end_logit[s_idx:], dim=0)
    return [s_idx, s_idx+e_idx]


def get_sentence_idx(first_word_locs, word_loc):
    """
    获取单词所在的句子索引。

    Args:
        first_word_locs: 每个句子第一个单词的位置列表
        word_loc: 当前单词的位置

    Returns:
        sent_idx: 单词所在的句子索引
    """
    sent_idx = -1
    for i, first_word_loc in enumerate(first_word_locs):
        if word_loc >= first_word_loc:
            sent_idx = i
        else:
            break
    return sent_idx


def get_maxtrix_value(X):
    """
    获取矩阵中最大值的索引。

    Args:
        X: 输入矩阵，形状为 [B, M, N]

    Returns:
        max_locs: 每个矩阵的最大值索引 [B, 2]
        cal_time: 计算时间
    """
    t1 = time.time()
    col_max, col_max_loc = X.max(dim=-1)
    _, row_max_loc = col_max.max(dim=-1)
    t2 = time.time()
    cal_time = (t2-t1)

    row_index = row_max_loc
    col_index = col_max_loc[torch.arange(row_max_loc.size(0)), row_index]

    return torch.stack((row_index, col_index)).T, cal_time


def get_best_indexes(features, feature_id_list, start_logit_list, end_logit_list, args):
    """
    获取最佳跨度的索引。

    Args:
        features: 特征列表
        feature_id_list: 特征 ID 列表
        start_logit_list: 起始位置 logits 列表
        end_logit_list: 结束位置 logits 列表
        args: 配置参数

    Returns:
        max_locs: 最佳跨度的索引
        cal_time, mask_time, score_time: 各阶段的计算时间
    """
    t1 = time.time()
    start_logits = torch.stack(tuple(start_logit_list)).unsqueeze(-1)  # [B, M, 1]
    end_logits = torch.stack(tuple(end_logit_list)).unsqueeze(1)  # [B, 1, M]
    scores = (start_logits + end_logits).float()
    t2 = time.time()
    score_time = t2 - t1

    def generate_mask(feature):
        """
        生成候选跨度的掩码。
        """
        mask = torch.zeros((args.max_enc_seq_length, args.max_enc_seq_length), dtype=float, device=args.device)
        context_length = len(feature.old_tok_to_new_tok_index)
        for start in range(context_length):
            start_index = feature.old_tok_to_new_tok_index[start][0]
            end_index_list = [feature.old_tok_to_new_tok_index[end-1][1] for end in range(start+1, min(context_length, start+args.max_span_length+1))]
            mask[start_index, end_index_list] = 1.0
        mask[0][0] = 1.0 
        return torch.log(mask).float().unsqueeze(0)
    
    t1 = time.time()
    candidate_masks = {feature_id: generate_mask(features[feature_id]) for feature_id in set(feature_id_list)}
    masks = torch.cat([candidate_masks[feature_id] for feature_id in feature_id_list], dim=0)

    t2 = time.time()
    mask_time = t2-t1
    masked_scores = scores + masks
    max_locs, cal_time = get_maxtrix_value(masked_scores)
    max_locs = [tuple(a) for a in max_locs]

    return max_locs, cal_time, mask_time, score_time


def get_best_index(feature, start_logit, end_logit, max_span_length, max_span_num, delta):
    """
    获取最佳跨度索引列表。

    Args:
        feature: 特征对象，包含旧到新 token 的索引映射。
        start_logit: 起始位置的 logits。
        end_logit: 结束位置的 logits。
        max_span_length: 最大跨度长度。
        max_span_num: 最大跨度数量。
        delta: 阈值偏移量。

    Returns:
        筛选后的最佳跨度列表。
    """
    th = start_logit[0] + end_logit[0]  # 初始阈值
    answer_span_list = []
    context_length = len(feature.old_tok_to_new_tok_index)

    # 遍历所有可能的跨度
    for start in range(context_length):
        for end in range(start + 1, min(context_length, start + max_span_length + 1)):
            start_index = feature.old_tok_to_new_tok_index[start][0]  # 起始 token 索引
            end_index = feature.old_tok_to_new_tok_index[end - 1][1]  # 结束 token 索引

            score = start_logit[start_index] + end_logit[end_index]  # 计算跨度分数
            answer_span = (start_index, end_index, score)

            # 如果分数超过阈值，添加到候选列表
            if score > (th + delta):
                answer_span_list.append(answer_span)

    # 如果没有找到合适的跨度，添加默认跨度
    if not answer_span_list:
        answer_span_list.append((0, 0, th))
    return filter_spans(answer_span_list, max_span_num)


def filter_spans(candidate_span_list, max_span_num):
    """
    筛选候选跨度，确保没有重叠并限制数量。

    Args:
        candidate_span_list: 候选跨度列表，每个跨度包含 (start, end, score)。
        max_span_num: 最大跨度数量。

    Returns:
        筛选后的跨度列表。
    """
    # 按分数从高到低排序
    candidate_span_list = sorted(candidate_span_list, key=lambda x: x[2], reverse=True)
    candidate_span_list = [(candidate_span[0], candidate_span[1]) for candidate_span in candidate_span_list]

    def is_intersect(span_1, span_2):
        """
        判断两个跨度是否有重叠。

        Args:
            span_1: 第一个跨度 (start, end)。
            span_2: 第二个跨度 (start, end)。

        Returns:
            是否有重叠。
        """
        return not (min(span_1[1], span_2[1]) < max(span_1[0], span_2[0]))

    if len(candidate_span_list) == 1:
        answer_span_list = candidate_span_list
    else:
        answer_span_list = []
        while candidate_span_list and len(answer_span_list) < max_span_num:
            selected_span = candidate_span_list[0]  # 选择分数最高的跨度
            answer_span_list.append(selected_span)
            candidate_span_list = candidate_span_list[1:]  # 移除已选择的跨度

            # 移除与已选择跨度有重叠的候选跨度
            candidate_span_list = [candidate_span for candidate_span in candidate_span_list if not is_intersect(candidate_span, selected_span)]
    return answer_span_list


def check_tensor(tensor, var_name):
    """
    打印张量的基本信息，用于调试。

    Args:
        tensor: 要检查的张量。
        var_name: 张量的变量名。
    """
    print("******Check*****")
    print("tensor_name: {}".format(var_name))
    print("shape: {}".format(tensor.size()))
    if len(tensor.size()) == 1 or tensor.size(0) <= 3:
        print("value: {}".format(tensor))
    else:
        print("part value: {}".format(tensor[0, :]))
    print("require_grads: {}".format(tensor.requires_grad))
    print("tensor_type: {}".format(tensor.dtype))


from spacy.tokens import Doc

class WhitespaceTokenizer:
    """
    自定义分词器，基于空格分词。
    """
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        """
        将文本分词为单词列表。

        Args:
            text: 输入文本。

        Returns:
            分词后的 Doc 对象。
        """
        words = text.split(" ")
        return Doc(self.vocab, words=words)


def find_head(arg_start, arg_end, doc):
    """
    找到指定范围内的头部单词。

    Args:
        arg_start: 参数的起始位置。
        arg_end: 参数的结束位置。
        doc: 文档对象。

    Returns:
        head_text: 头部单词的文本。
    """
    arg_end -= 1  # 调整结束位置
    cur_i = arg_start
    while doc[cur_i].head.i >= arg_start and doc[cur_i].head.i <= arg_end:
        if doc[cur_i].head.i == cur_i:
            # 当前单词是头部
            break
        else:
            cur_i = doc[cur_i].head.i  # 更新当前索引为头部索引

    arg_head = cur_i
    head_text = doc[arg_head]
    return head_text