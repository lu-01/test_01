import sys
sys.path.append("../")  # 添加上级目录到系统路径，方便模块导入
import copy
import spacy

from utils import _normalize_answer, find_head, hungarian_matcher
from utils import WhitespaceTokenizer
nlp = spacy.load('en_core_web_sm')  # 加载 SpaCy 的英文模型
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)  # 使用自定义的空白分词器


def eval_rpf(gt_num, pred_num, correct_num):
    """
    计算召回率、精确率和 F1 分数。

    Args:
        gt_num: 真实值数量。
        pred_num: 预测值数量。
        correct_num: 正确匹配的数量。

    Returns:
        res: 包含召回率、精确率、F1 分数和数量统计的字典。
    """
    recall = correct_num / gt_num if gt_num != 0 else .0
    precision = correct_num / pred_num if pred_num != 0 else .0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 1e-4 else .0
    res = {
        "recall": recall, "precision": precision, "f1": f1,
        "gt_num": gt_num, "pred_num": pred_num, "correct_num": correct_num,
    }
    return res


def eval_std_f1_score(features, invalid_gt_num=0):
    """
    计算标准 F1 分数。

    Args:
        features: 特征列表。
        invalid_gt_num: 无效的真实值数量。

    Returns:
        res_classification: 分类任务的评估结果。
        res_identification: 识别任务的评估结果。
    """
    gt_num, pred_num, correct_num = 0, 0, 0
    gt_num_identify, pred_num_identify, correct_identify_num = 0, 0, 0
    
    for feature in features:
        all_pred_list = list()
        all_gt_list = list()
        for role in feature.arg_list:
            gt_list = feature.gt_dict_word[role] if role in feature.gt_dict_word else list()
            pred_list = list(set(feature.pred_dict_word[role])) if role in feature.pred_dict_word else list()
            gt_num += len(gt_list)
            pred_num += len(pred_list)
            
            for gt_span in gt_list:
                if gt_span in pred_list:
                    correct_num += 1

            all_pred_list.extend(copy.deepcopy(pred_list))
            all_gt_list.extend(gt_list)

        all_pred_list = list(set(all_pred_list))
        all_gt_list = list(set(all_gt_list))
        pred_num_identify += len(all_pred_list)
        gt_num_identify += len(all_gt_list)
        for gt_span in all_gt_list:
            if gt_span in all_pred_list:
                correct_identify_num += 1
        
    res_classification = eval_rpf(gt_num + invalid_gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify + invalid_gt_num, pred_num_identify, correct_identify_num)
    return res_classification, res_identification


def eval_text_f1_score(features, invalid_gt_num=0):
    """
    计算基于文本的 F1 分数。

    Args:
        features: 特征列表。
        invalid_gt_num: 无效的真实值数量。

    Returns:
        res_classification: 分类任务的评估结果。
        res_identification: 识别任务的评估结果。
    """
    gt_num, pred_num, correct_num = 0, 0, 0
    gt_num_identify, pred_num_identify, correct_identify_num = 0, 0, 0

    for feature in features:
        all_pred_list = list()
        all_gt_list = list()
        full_text = feature.full_text
        for role in feature.arg_list:
            gt_list = feature.gt_dict_word[role] if role in feature.gt_dict_word else list()
            pred_list = list(set(feature.pred_dict_word[role])) if role in feature.pred_dict_word else list()
            # 将跨度转换为标准化的文本
            gt_texts = [_normalize_answer(" ".join(full_text[gt_span[0]: gt_span[1] + 1])) for gt_span in gt_list]
            pred_texts = list(set([_normalize_answer(" ".join(full_text[pred_span[0]: pred_span[1] + 1])) for pred_span in copy.deepcopy(pred_list)]))
            gt_list = gt_texts
            pred_list = pred_texts
            
            gt_num += len(gt_list)
            pred_num += len(pred_list)
            
            for gt_span in gt_list:
                if gt_span in pred_list:
                    correct_num += 1

            all_pred_list.extend(copy.deepcopy(pred_list))
            all_gt_list.extend(gt_list)

        all_pred_list = list(set(all_pred_list))
        all_gt_list = list(set(all_gt_list))
        pred_num_identify += len(all_pred_list)
        gt_num_identify += len(all_gt_list)
        for gt_span in all_gt_list:
            if gt_span in all_pred_list:
                correct_identify_num += 1
        
    res_classification = eval_rpf(gt_num + invalid_gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify + invalid_gt_num, pred_num_identify, correct_identify_num)
    return res_classification, res_identification


def eval_head_f1_score(features, invalid_gt_num=0):
    """
    计算基于头部的 F1 分数。

    Args:
        features: 特征列表。
        invalid_gt_num: 无效的真实值数量。

    Returns:
        res_classification: 分类任务的评估结果。
        res_identification: 识别任务的评估结果。
    """
    gt_num, pred_num, correct_num = 0, 0, 0
    gt_num_identify, pred_num_identify, correct_identify_num = 0, 0, 0
    last_full_text = None

    for feature in features:
        all_pred_list = list()
        all_gt_list = list()
        full_text = feature.full_text
        for role in feature.arg_list:
            gt_list = feature.gt_dict_word[role] if role in feature.gt_dict_word else list()
            pred_list = list(set(feature.pred_dict_word[role])) if role in feature.pred_dict_word else list()
            # 使用 SpaCy 提取头部信息
            if full_text != last_full_text:
                doc = nlp(" ".join(full_text))  # 生成 SpaCy 文档
                last_full_text = full_text

            gt_head_texts = [str(find_head(gt_span[0], gt_span[1] + 1, doc)) for gt_span in gt_list]
            pred_head_texts = list(set([str(find_head(pred_span[0], pred_span[1] + 1, doc)) for pred_span in copy.deepcopy(pred_list)]))
            gt_list = gt_head_texts
            pred_list = pred_head_texts
            
            gt_num += len(gt_list)
            pred_num += len(pred_list)
            
            for gt_span in gt_list:
                if gt_span in pred_list:
                    correct_num += 1

            all_pred_list.extend(copy.deepcopy(pred_list))
            all_gt_list.extend(gt_list)

        all_pred_list = list(set(all_pred_list))
        all_gt_list = list(set(all_gt_list))
        pred_num_identify += len(all_pred_list)
        gt_num_identify += len(all_gt_list)
        for gt_span in all_gt_list:
            if gt_span in all_pred_list:
                correct_identify_num += 1
        
    res_classification = eval_rpf(gt_num + invalid_gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify + invalid_gt_num, pred_num_identify, correct_identify_num)
    return res_classification, res_identification


def show_results(features, output_file, metainfo):
    """
    显示评估结果并将其写入文件。

    Args:
        features: 特征列表。
        output_file: 输出文件路径。
        metainfo: 元信息字典。
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for k, v in metainfo.items():
            f.write(f"{k}: {v}\n")

        for feature in features:
            example_id = feature.example_id
            sent = feature.enc_text
            f.write("-------------------------------------------------------------------------------------\n")
            f.write("Sent: {}\n".format(sent))
            f.write("Event type: {}\t\t\tTrigger word: {}\n".format(feature.event_type, feature.event_trigger))
            f.write("Example ID {}\n".format(example_id))
            full_text = feature.full_text
            for arg_role in feature.arg_list:
                pred_list = feature.pred_dict_word[arg_role] if arg_role in feature.pred_dict_word else list()
                gt_list = feature.gt_dict_word[arg_role] if arg_role in feature.gt_dict_word else list()
                if len(pred_list) == 0 and len(gt_list) == 0:
                    continue
                
                if len(gt_list) == 0 and len(pred_list) > 0:
                    gt_list = [(-1, -1)] * len(pred_list)
                
                if len(gt_list) > 0 and len(pred_list) == 0:
                    pred_list = [(-1, -1)] * len(gt_list)

                gt_idxs, pred_idxs = hungarian_matcher(gt_list, pred_list)

                for pred_idx, gt_idx in zip(pred_idxs, gt_idxs):
                    if gt_list[gt_idx] == (-1, -1) and pred_list[pred_idx] == (-1, -1):
                        continue
                    else:
                        pred_text = " ".join(full_text[pred_list[pred_idx][0]: pred_list[pred_idx][1] + 1]) if pred_list[pred_idx] != (-1, -1) else "__ No answer __"
                        gt_text = " ".join(full_text[gt_list[gt_idx][0]: gt_list[gt_idx][1] + 1]) if gt_list[gt_idx] != (-1, -1) else "__ No answer __"
                    
                    if gt_list[gt_idx] == pred_list[pred_idx]:
                        f.write("Arg {} matched: Pred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role, pred_text, pred_list[pred_idx][0], pred_list[pred_idx][1], gt_text, gt_list[gt_idx][0], gt_list[gt_idx][1]))
                    else:
                        f.write("Arg {} dismatched: Pred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role, pred_text, pred_list[pred_idx][0], pred_list[pred_idx][1], gt_text, gt_list[gt_idx][0], gt_list[gt_idx][1]))
                
                if len(gt_idxs) < len(gt_list):  # 预测中缺少答案
                    for idx in range(len(gt_list)):
                        if idx not in gt_idxs:
                            gt_text = " ".join(full_text[gt_list[idx][0]: gt_list[idx][1] + 1])
                            f.write("Arg {} dismatched: Pred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role, "__ No answer __", -1, -1, gt_text, gt_list[idx][0], gt_list[idx][1])) 

                if len(pred_idxs) < len(pred_list):  # 真实值中缺少答案
                    for idx in range(len(pred_list)):
                        if idx not in pred_idxs:
                            pred_text = " ".join(full_text[pred_list[idx][0]: pred_list[idx][1] + 1])
                            f.write("Arg {} dismatched: Pred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role, pred_text, pred_list[idx][0], pred_list[idx][1], "__ No answer __", -1, -1))