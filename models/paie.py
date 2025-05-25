# paie model
import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel
from utils import hungarian_matcher, get_best_span, get_best_span_simple


class PAIE(BartPretrainedModel):
    """
    PAIE 模型类，基于 BART 的预训练模型，用于多提示事件抽取任务。
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = BartModel(config)  # 初始化 BART 模型
        self.w_prompt_start = nn.Parameter(torch.rand(config.d_model, ))  # 起始位置提示权重
        self.w_prompt_end = nn.Parameter(torch.rand(config.d_model, ))  # 结束位置提示权重

        # 初始化权重
        self.model._init_weights(self.w_prompt_start)
        self.model._init_weights(self.w_prompt_end)
        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')  # 定义交叉熵损失函数

    def forward(
        self,
        enc_input_ids=None,
        enc_mask_ids=None,
        dec_prompt_ids=None,
        dec_prompt_mask_ids=None,
        arg_joint_prompts=None,
        target_info=None,
        old_tok_to_new_tok_indexs=None,
        arg_list=None,
    ):
        """
        前向传播函数。

        Args:
            enc_input_ids: 编码器输入 ID。
            enc_mask_ids: 编码器掩码 ID。
            dec_prompt_ids: 解码器提示 ID。
            dec_prompt_mask_ids: 解码器提示掩码 ID。
            arg_joint_prompts: 参数联合提示。
            target_info: 目标信息（仅在训练时使用）。
            old_tok_to_new_tok_indexs: 旧到新 token 的索引映射。
            arg_list: 参数列表。

        Returns:
            如果是训练模式，返回总损失和 logits 列表；
            如果是推理模式，返回空列表和 logits 列表。
        """
        # 根据配置选择上下文表示方式（编码器或解码器）
        if self.config.context_representation == 'decoder':
            context_outputs = self.model(
                enc_input_ids,
                attention_mask=enc_mask_ids,
                return_dict=True,
            )
            decoder_context = context_outputs.encoder_last_hidden_state
            context_outputs = context_outputs.last_hidden_state
        else:
            context_outputs = self.model.encoder(
                enc_input_ids,
                attention_mask=enc_mask_ids,
            )
            context_outputs = context_outputs.last_hidden_state
            decoder_context = context_outputs

        # 解码器处理提示信息
        decoder_prompt_outputs = self.model.decoder(
            input_ids=dec_prompt_ids,
            attention_mask=dec_prompt_mask_ids,
            encoder_hidden_states=decoder_context,
            encoder_attention_mask=enc_mask_ids,
        )
        decoder_prompt_outputs = decoder_prompt_outputs.last_hidden_state  # [bs, prompt_len, H]

        logit_lists = list()  # 存储所有参数角色的 logits
        total_loss = 0.  # 初始化总损失

        # 遍历每个样本
        for i, (context_output, decoder_prompt_output, arg_joint_prompt, old_tok_to_new_tok_index) in \
            enumerate(zip(context_outputs, decoder_prompt_outputs, arg_joint_prompts, old_tok_to_new_tok_indexs)):
            
            batch_loss = list()  # 存储当前样本的损失
            cnt = 0  # 计数器，用于计算平均损失
            
            output = dict()  # 存储当前样本的预测结果
            for arg_role in arg_joint_prompt.keys():
                """
                arg_role: 参数角色，例如 "Agent" 或 "Patient"。
                arg_joint_prompt[arg_role]: 包含起始和结束位置的提示信息。
                """
                prompt_slots = arg_joint_prompt[arg_role]

                start_logits_list = list()  # 存储起始位置 logits
                end_logits_list = list()  # 存储结束位置 logits

                # 遍历提示槽，计算每个提示的 logits
                for (p_start, p_end) in zip(prompt_slots['tok_s'], prompt_slots['tok_e']):
                    prompt_query_sub = decoder_prompt_output[p_start:p_end]
                    prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0)  # 平均池化提示向量
                    
                    start_query = (prompt_query_sub * self.w_prompt_start).unsqueeze(-1)  # 起始位置查询向量
                    end_query = (prompt_query_sub * self.w_prompt_end).unsqueeze(-1)  # 结束位置查询向量

                    start_logits = torch.bmm(context_output.unsqueeze(0), start_query).squeeze()  # 计算起始位置 logits
                    end_logits = torch.bmm(context_output.unsqueeze(0), end_query).squeeze()  # 计算结束位置 logits
                    
                    start_logits_list.append(start_logits)
                    end_logits_list.append(end_logits)
                    
                output[arg_role] = [start_logits_list, end_logits_list]  # 保存当前参数角色的 logits

                if self.training:
                    # 计算损失
                    target = target_info[i][arg_role]  # 获取目标信息
                    predicted_spans = list()

                    # 根据配置选择匹配方法
                    for (start_logits, end_logits) in zip(start_logits_list, end_logits_list):
                        if self.config.matching_method_train == 'accurate':
                            predicted_spans.append(get_best_span(start_logits, end_logits, old_tok_to_new_tok_index, self.config.max_span_length))
                        elif self.config.matching_method_train == 'max':
                            predicted_spans.append(get_best_span_simple(start_logits, end_logits))
                        else:
                            raise AssertionError()

                    target_spans = [[s, e] for (s, e) in zip(target["span_s"], target["span_e"])]
                    if len(target_spans) < len(predicted_spans):
                        # 如果目标跨度数量少于预测跨度，进行填充
                        pad_len = len(predicted_spans) - len(target_spans)
                        target_spans = target_spans + [[0, 0]] * pad_len
                        target["span_s"] = target["span_s"] + [0] * pad_len
                        target["span_e"] = target["span_e"] + [0] * pad_len
                        
                    # 使用匈牙利算法或简单匹配进行跨度匹配
                    if self.config.bipartite:
                        idx_preds, idx_targets = hungarian_matcher(predicted_spans, target_spans)
                    else:
                        idx_preds = list(range(len(predicted_spans)))
                        idx_targets = list(range(len(target_spans)))
                        if len(idx_targets) > len(idx_preds):
                            idx_targets = idx_targets[0:len(idx_preds)]
                        idx_preds = torch.as_tensor(idx_preds, dtype=torch.int64)
                        idx_targets = torch.as_tensor(idx_targets, dtype=torch.int64)

                    cnt += len(idx_preds)  # 更新计数器
                    # 计算起始和结束位置的损失
                    start_loss = self.loss_fct(torch.stack(start_logits_list)[idx_preds], torch.LongTensor(target["span_s"]).to(self.config.device)[idx_targets])
                    end_loss = self.loss_fct(torch.stack(end_logits_list)[idx_preds], torch.LongTensor(target["span_e"]).to(self.config.device)[idx_targets])
                    batch_loss.append((start_loss + end_loss) / 2)  # 平均损失
                
            logit_lists.append(output)  # 保存当前样本的预测结果
            if self.training:  # 计算当前批次的平均损失
                total_loss = total_loss + torch.sum(torch.stack(batch_loss)) / cnt
            
        if self.training:
            return total_loss / len(context_outputs), logit_lists  # 返回平均损失和 logits 列表
        else:
            return [], logit_lists  # 推理模式下返回空损失和 logits 列表