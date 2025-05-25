import torch.nn as nn
import logging
logger = logging.getLogger(__name__)  # 设置日志记录器


class BaseTrainer:
    """
    基础训练器类，提供通用的训练功能。
    """
    def __init__(
        self,
        cfg=None,  # 配置对象，包含训练参数
        data_loader=None,  # 数据加载器，用于提供训练数据
        model=None,  # 模型对象
        optimizer=None,  # 优化器
        scheduler=None,  # 学习率调度器
    ):
        self.cfg = cfg
        self.data_loader = data_loader
        self.data_iterator = iter(self.data_loader)  # 初始化数据迭代器
        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler
        self._init_metric()  # 初始化训练指标

    def _init_metric(self):
        """
        初始化训练指标。
        """
        self.metric = {
            "global_steps": 0,  # 全局训练步数
            "smooth_loss": 0.0,  # 平滑损失，用于日志记录
        }

    def write_log(self):
        """
        记录训练日志，包括当前步数、学习率和损失。
        """
        logger.info("-----------------------global_step: {} -------------------------------- ".format(self.metric['global_steps']))
        logger.info('lr: {}'.format(self.scheduler.get_last_lr()[0]))  # 当前学习率
        logger.info('smooth_loss: {}'.format(self.metric['smooth_loss']))  # 平滑损失
        self.metric['smooth_loss'] = 0.0  # 重置平滑损失

    def train_one_step(self):
        """
        执行单步训练，包括前向传播、反向传播和参数更新。
        """
        self.model.train()  # 设置模型为训练模式
        try:
            batch = next(self.data_iterator)  # 获取下一个批次数据
        except StopIteration:
            # 如果数据迭代器耗尽，重新初始化迭代器
            self.data_iterator = iter(self.data_loader)
            batch = next(self.data_iterator)

        # 将批次数据转换为模型输入
        inputs = self.convert_batch_to_inputs(batch)
        loss, _ = self.model(**inputs)  # 前向传播计算损失

        # 如果使用梯度累积，调整损失值
        if self.cfg.gradient_accumulation_steps > 1:
            loss = loss / self.cfg.gradient_accumulation_steps
        loss.backward()  # 反向传播计算梯度

        # 如果设置了梯度裁剪，执行梯度裁剪
        if self.cfg.max_grad_norm != 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        
        # 累加平滑损失，用于日志记录
        self.metric['smooth_loss'] += loss.item() / self.cfg.logging_steps

        # 每完成一次梯度累积，更新参数
        if (self.metric['global_steps'] + 1) % self.cfg.gradient_accumulation_steps == 0:
            self.optimizer.step()  # 更新参数
            self.scheduler.step()  # 更新学习率
            self.model.zero_grad()  # 清空梯度
            self.metric['global_steps'] += 1  # 更新全局步数

    def convert_batch_to_inputs(self, batch):
        """
        将批次数据转换为模型输入（需要子类实现）。
        """
        raise NotImplementedError()


class Trainer(BaseTrainer):
    """
    具体训练器类，继承自 BaseTrainer，提供特定模型的输入转换逻辑。
    """
    def __init__(self, cfg=None, data_loader=None, model=None, optimizer=None, scheduler=None):
        super().__init__(cfg, data_loader, model, optimizer, scheduler)

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
                'target_info': batch[6],  # 目标信息
                'old_tok_to_new_tok_indexs': batch[7],  # 旧到新 token 的索引映射
                'arg_joint_prompts': batch[8],  # 参数联合提示
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
                'start_position_ids': [item.to(self.cfg.device) for item in batch[14]],  # 起始位置 ID
                'end_position_ids': [item.to(self.cfg.device) for item in batch[15]],  # 结束位置 ID
            }

        return inputs  # 返回模型输入