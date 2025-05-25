import torch
import logging
logger = logging.getLogger(__name__)

from utils import get_best_indexes, get_best_index


class BaseEvaluator:
    """
    基础评估器类，提供通用的评估功能。
    """
    def __init__(
        self,
        cfg=None,  # 配置对象，包含评估参数
        data_loader=None,  # 数据加载器，用于提供评估数据
        model=None,  # 模型对象
        metric_fn_dict=None,  # 评估指标函数字典
    ):
        self.cfg = cfg
        self.eval_loader = data_loader
        self.model = model
        self.metric_fn_dict = metric_fn_dict

    def _init_metric(self):
        """
        初始化评估指标字典。
        """
        self.metric_val_dict = {metric: None for metric in self.metric_fn_dict}

    def calculate_one_batch(self, batch):
        """
        计算单个批次的模型输出。
        """
        inputs, named_v = self.convert_batch_to_inputs(batch)  # 转换批次数据为模型输入
        with torch.no_grad():  # 禁用梯度计算
            _, outputs_list = self.model(**inputs)  # 获取模型输出
        return outputs_list, named_v

    def evaluate_one_batch(self, batch):
        """
        评估单个批次数据。
        """
        outputs_list, named_v = self.calculate_one_batch(batch)  # 计算模型输出
        self.collect_fn(outputs_list, named_v, batch)  # 收集评估结果

    def evaluate(self):
        """
        执行完整的评估过程。
        """
        self.model.eval()  # 设置模型为评估模式
        self.build_and_clean_record()  # 初始化记录
        self._init_metric()  # 初始化评估指标
        for batch in self.eval_loader:  # 遍历评估数据
            self.evaluate_one_batch(batch)  # 评估每个批次
        output = self.predict()  # 生成预测结果
        return output

    def build_and_clean_record(self):
        """
        初始化或清理记录（需要子类实现）。
        """
        raise NotImplementedError()

    def collect_fn(self, outputs_list, named_v, batch):
        """
        收集评估结果（需要子类实现）。
        """
        raise NotImplementedError()

    def convert_batch_to_inputs(self, batch):
        """
        转换批次数据为模型输入（需要子类实现）。
        """
        raise NotImplementedError()

    def predict(self):
        """
        生成预测结果（需要子类实现）。
        """
        raise NotImplementedError()


class Evaluator(BaseEvaluator):
    """
    具体评估器类，继承自 BaseEvaluator，提供特定模型的评估逻辑。
    """
    def __init__(
        self, 
        cfg=None,  # 配置对象
        data_loader=None,  # 数据加载器
        model=None,  # 模型对象
        metric_fn_dict=None,  # 评估指标函数字典
        features=None,  # 特征列表
        set_type=None,  # 数据集类型（如验证集或测试集）
        invalid_num=0,  # 无效样本数量
    ):
        super().__init__(cfg, data_loader, model, metric_fn_dict)
        self.features = features
        self.set_type = set_type
        self.invalid_num = invalid_num

    def convert_batch_to_inputs(self, batch):
        """
        根据模型类型，将批次数据转换为模型输入。
        """
        if self.cfg.model_type == "paie":  # 如果模型类型为 "paie"
            inputs = {
                'enc_input_ids':  batch[0].to(self.cfg.device),  # 编码器输入 ID
                'enc_mask_ids':   batch[1].to(self.cfg.device),  # 编码器掩码 ID
                'dec_prompt_ids': batch[4].to(self.cfg.device),  # 解码器提示 ID
                'dec_prompt_mask_ids': batch[5].to(self.cfg.device),  # 解码器提示掩码 ID
                'old_tok_to_new_tok_indexs': batch[7],  # 旧到新 token 的索引映射
                'arg_joint_prompts': batch[8],  # 参数联合提示
                'target_info': None,  # 目标信息（评估时不需要）
                'arg_list': batch[9],  # 参数列表
            }
        elif self.cfg.model_type == "base":  # 如果模型类型为 "base"
            inputs = {
                'enc_input_ids':  batch[0].to(self.cfg.device),  # 编码器输入 ID
                'enc_mask_ids':   batch[1].to(self.cfg.device),  # 编码器掩码 ID
                'decoder_prompt_ids_list': [item.to(self.cfg.device) for item in batch[2]],  # 解码器提示 ID 列表
                'decoder_prompt_mask_list': [item.to(self.cfg.device) for item in batch[3]],  # 解码器提示掩码列表
                'arg_list': batch[9],  # 参数列表
                'decoder_prompt_start_positions_list': [item.to(self.cfg.device) for item in batch[12]],  # 解码器提示起始位置列表
                'decoder_prompt_end_positions_list': [item.to(self.cfg.device) for item in batch[13]],  # 解码器提示结束位置列表
            }

        named_v = {
            "arg_roles": batch[9],  # 参数角色
            "feature_ids": batch[11],  # 特征 ID
        }
        return inputs, named_v

    def build_and_clean_record(self):
        """
        初始化记录字典，用于存储评估过程中生成的中间结果。
        """
        self.record = {
            "feature_id_list": list(),  # 特征 ID 列表
            "role_list": list(),  # 参数角色列表
            "full_start_logit_list": list(),  # 起始位置 logits 列表
            "full_end_logit_list": list(),  # 结束位置 logits 列表
        }

    def collect_fn(self, outputs_list, named_v, batch):
        """
        收集模型输出并存储到记录中。
        """
        bs = len(batch[0])  # 批次大小
        for i in range(bs):
            predictions = outputs_list[i]  # 获取当前样本的预测结果
            feature_id = named_v["feature_ids"][i].item()  # 获取特征 ID
            for arg_role in named_v["arg_roles"][i]:  # 遍历参数角色
                [start_logits_list, end_logits_list] = predictions[arg_role]  # 获取起始和结束 logits
                for (start_logit, end_logit) in zip(start_logits_list, end_logits_list):
                    self.record["feature_id_list"].append(feature_id)
                    self.record["role_list"].append(arg_role)
                    self.record["full_start_logit_list"].append(start_logit)
                    self.record["full_end_logit_list"].append(end_logit)

    def predict(self):
        """
        根据记录生成最终的预测结果。
        """
        # 初始化特征的预测结果
        for feature in self.features:
            feature.init_pred()
            feature.set_gt(self.cfg.model_type, self.cfg.dataset_type)

        if self.cfg.model_type == 'paie':  # 如果模型类型为 "paie"
            pred_list = []
            # 批量处理 logits，获取最佳预测位置
            for s in range(0, len(self.record["full_start_logit_list"]), self.cfg.infer_batch_size):
                sub_max_locs, cal_time, mask_time, score_time = get_best_indexes(
                    self.features,
                    self.record["feature_id_list"][s:s+self.cfg.infer_batch_size],
                    self.record["full_start_logit_list"][s:s+self.cfg.infer_batch_size],
                    self.record["full_end_logit_list"][s:s+self.cfg.infer_batch_size],
                    self.cfg
                )
                pred_list.extend(sub_max_locs)
            # 将预测结果添加到对应的特征中
            for (pred, feature_id, role) in zip(pred_list, self.record["feature_id_list"], self.record["role_list"]):
                pred_span = (pred[0].item(), pred[1].item())  # 转换为预测跨度
                feature = self.features[feature_id]
                feature.add_pred(role, pred_span, self.cfg.dataset_type)
        else:  # 如果模型类型为 "base"
            for feature_id, role, start_logit, end_logit in zip(
                self.record["feature_id_list"], self.record["role_list"], self.record["full_start_logit_list"], self.record["full_end_logit_list"]
            ):
                feature = self.features[feature_id]
                # 获取最佳预测跨度
                answer_span_pred_list = get_best_index(
                    feature, start_logit, end_logit,
                    max_span_length=self.cfg.max_span_length,
                    max_span_num=int(self.cfg.max_span_num_dict[feature.event_type][role]),
                    delta=self.cfg.th_delta
                )
                # 将预测结果添加到对应的特征中
                for pred_span in answer_span_pred_list:
                    feature.add_pred(role, pred_span, self.cfg.dataset_type)

        # 计算评估指标
        for metric, eval_fn in self.metric_fn_dict.items():
            perf_c, perf_i = eval_fn(self.features, self.invalid_num)  # 分类和识别性能
            self.metric_val_dict[metric] = (perf_c, perf_i)
            logger.info('{}-Classification. {} ({}): R {} P {} F {}'.format(
                metric, self.set_type, perf_c['gt_num'], perf_c['recall'], perf_c['precision'], perf_c['f1']))
            logger.info('{}-Identification. {} ({}): R {} P {} F {}'.format(
                metric, self.set_type, perf_i['gt_num'], perf_i['recall'], perf_i['precision'], perf_i['f1']))

        return self.metric_val_dict['span']  # 返回 "span" 指标的评估结果