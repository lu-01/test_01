import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel


class BartSingleArg(BartPretrainedModel):
    """
    基于 BART 的单参数事件抽取模型。
    """
    def __init__(self, config):
        """
        初始化模型。

        Args:
            config: 模型配置对象。
        """
        super().__init__(config)
        self.config = config
        self.model = BartModel(config)  # 初始化 BART 模型
        self.w_prompt_start = nn.Parameter(torch.rand(config.d_model, ))  # 起始位置提示权重
        self.w_prompt_end = nn.Parameter(torch.rand(config.d_model, ))  # 结束位置提示权重

        # 初始化权重
        self.model._init_weights(self.w_prompt_start)
        self.model._init_weights(self.w_prompt_end)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='sum')  # 二分类交叉熵损失函数
        self.logsoft_fct = nn.LogSoftmax(dim=-1)  # LogSoftmax 函数

    def forward(
        self,
        enc_input_ids=None,
        enc_mask_ids=None,
        decoder_prompt_ids_list=None,
        decoder_prompt_mask_list=None,
        decoder_prompt_start_positions_list=None,
        decoder_prompt_end_positions_list=None,
        start_position_ids=None,
        end_position_ids=None,
        arg_list=None,
    ):
        """
        前向传播函数。

        Args:
            enc_input_ids: 编码器输入 ID。
            enc_mask_ids: 编码器掩码 ID。
            decoder_prompt_ids_list: 解码器提示 ID 列表。
            decoder_prompt_mask_list: 解码器提示掩码列表。
            decoder_prompt_start_positions_list: 解码器提示起始位置列表。
            decoder_prompt_end_positions_list: 解码器提示结束位置列表。
            start_position_ids: 起始位置的真实标签。
            end_position_ids: 结束位置的真实标签。
            arg_list: 参数角色列表。

        Returns:
            如果是训练模式，返回平均损失和 logits 列表；
            如果是推理模式，返回空列表和 logits 列表。
        """
        # 编码器前向传播
        context_outputs = self.model(
            enc_input_ids,
            attention_mask=enc_mask_ids,
            return_dict=True,
        )
        context_encoder_outputs = context_outputs.encoder_last_hidden_state  # 编码器的最后隐藏状态
        context_decoder_outputs = context_outputs.last_hidden_state  # 解码器的最后隐藏状态

        logit_lists = list()  # 存储所有参数角色的 logits
        total_loss = list()  # 存储所有样本的损失

        # 遍历每个样本
        for i, (decoder_prompt_ids, decoder_prompt_mask, decoder_prompt_start_positions, decoder_prompt_end_positions) in \
            enumerate(zip(decoder_prompt_ids_list, decoder_prompt_mask_list, decoder_prompt_start_positions_list, decoder_prompt_end_positions_list)):

            # 解码器处理提示信息
            prompt_decoder_outputs = self.model.decoder(
                input_ids=decoder_prompt_ids,
                attention_mask=decoder_prompt_mask,
                encoder_hidden_states=context_encoder_outputs[i:i+1].repeat(decoder_prompt_ids.size(0), 1, 1),
                encoder_attention_mask=enc_mask_ids[i:i+1].repeat(decoder_prompt_ids.size(0), 1),
            )
            prompt_decoder_outputs = prompt_decoder_outputs.last_hidden_state  # [Arg_num, Query_L, H]

            # 计算提示向量
            for j, (p_start, p_end, prompt_decoder_output) in enumerate(zip(decoder_prompt_start_positions, decoder_prompt_end_positions, prompt_decoder_outputs)):
                prompt_query_sub = prompt_decoder_output[p_start:p_end]
                prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0)  # 平均池化提示向量
                prompt_query = torch.cat((prompt_query, prompt_query_sub), dim=0) if j > 0 else prompt_query_sub
            
            # 计算起始和结束位置的提示查询向量
            start_prompt_query = (prompt_query * self.w_prompt_start[None, :]).unsqueeze(-1)  # [Arg_num, H, 1]
            end_prompt_query = (prompt_query * self.w_prompt_end[None, :]).unsqueeze(-1)  # [Arg_num, H, 1]

            # 计算起始和结束位置的 logits
            start_logits = torch.bmm(context_decoder_outputs[i:i+1].repeat(len(start_prompt_query), 1, 1), start_prompt_query).squeeze(-1)  # [Arg_num, L]
            end_logits = torch.bmm(context_decoder_outputs[i:i+1].repeat(len(end_prompt_query), 1, 1), end_prompt_query).squeeze(-1)
            start_logits = start_logits.masked_fill_(~enc_mask_ids[i:i+1].repeat(len(start_prompt_query), 1).bool(), -20)  # 掩码处理
            end_logits = end_logits.masked_fill_(~enc_mask_ids[i:i+1].repeat(len(start_prompt_query), 1).bool(), -20)

            # 计算损失（仅在训练模式下）
            if start_position_ids is not None and end_position_ids is not None:
                start_logsoftmax = self.logsoft_fct(start_logits)
                end_logsoftmax = self.logsoft_fct(end_logits)
                start_loss = -torch.mean(torch.sum(start_position_ids[i] * start_logsoftmax, dim=1), dim=0)
                end_loss = -torch.mean(torch.sum(end_position_ids[i] * end_logsoftmax, dim=1), dim=0)
                total_loss.append((start_loss + end_loss) / 2)

            # 保存当前样本的预测结果
            output = dict()
            for j, arg_role in enumerate(arg_list[i]):
                start_logits_list, end_logits_list = [start_logits[j]], [end_logits[j]]
                output[arg_role] = [start_logits_list, end_logits_list]
            logit_lists.append(output)

        # 返回损失和 logits 列表
        if total_loss:
            return torch.mean(torch.stack(total_loss)), logit_lists  # 返回平均损失和 logits 列表
        else:
            return [], logit_lists  # 推理模式下返回空损失和 logits 列表