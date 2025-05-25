import os
import sys
sys.path.append("../")  # 添加上级目录到系统路径，方便模块导入
import logging
logger = logging.getLogger(__name__)  # 设置日志记录器

from metric import eval_std_f1_score, eval_text_f1_score, eval_head_f1_score, show_results  # 导入评估函数和结果展示函数
from runner.train import Trainer  # 导入训练器类
from runner.evaluate import Evaluator  # 导入评估器类


class BaseRunner:
    """
    基础运行器类，提供训练和推理的基本功能。
    """
    def __init__(
        self,
        cfg=None,  # 配置对象，包含运行参数
        data_samples=None,  # 数据样本（训练、验证、测试）
        data_features=None,  # 数据特征（训练、验证、测试）
        data_loaders=None,  # 数据加载器（训练、验证、测试）
        model=None,  # 模型对象
        optimizer=None,  # 优化器
        scheduler=None,  # 学习率调度器
        metric_fn_dict=None,  # 评估指标函数字典
    ):
        # 初始化类属性
        self.cfg = cfg
        self.model = model
        self.train_samples, self.dev_samples, self.test_samples = data_samples
        self.train_features, self.dev_features, self.test_features = data_features
        self.train_loader, self.dev_loader, self.test_loader = data_loaders

        # 初始化训练器
        self.trainer = Trainer(
            cfg=self.cfg,
            data_loader=self.train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        # 初始化验证集评估器
        self.dev_evaluator = Evaluator(
            cfg=self.cfg,
            data_loader=self.dev_loader,
            model=model,
            metric_fn_dict=metric_fn_dict,
            features=self.dev_features,
            set_type="DEV",  # 设置评估类型为验证集
            invalid_num=self.cfg.dev_invalid_num,  # 无效样本数量
        )
        # 初始化测试集评估器
        self.test_evaluator = Evaluator(
            cfg=self.cfg,
            data_loader=self.test_loader,
            model=model,
            metric_fn_dict=metric_fn_dict,
            features=self.test_features,
            set_type="TEST",  # 设置评估类型为测试集
            invalid_num=self.cfg.test_invalid_num,  # 无效样本数量
        )

    def run(self):
        """
        根据配置决定运行模式（训练或推理）。
        """
        if self.cfg.inference_only:
            self.inference()  # 仅推理模式
        else:
            self.train()  # 训练模式

    def train(self):
        """
        执行训练过程。
        """
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_loader) * self.cfg.batch_size)
        logger.info("  batch size = %d", self.cfg.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.cfg.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.cfg.max_steps)

        for global_step in range(self.cfg.max_steps):
            self.trainer.train_one_step()  # 执行单步训练

            if (global_step + 1) % self.cfg.logging_steps == 0:
                self.trainer.write_log()  # 定期记录日志

            if (global_step + 1) % self.cfg.eval_steps == 0:
                self.eval_and_update(global_step)  # 定期评估并更新

    def inference(self):
        """
        执行推理过程。
        """
        dev_c, _ = self.dev_evaluator.evaluate()  # 验证集评估
        test_c, _ = self.test_evaluator.evaluate()  # 测试集评估
        self.report_result(dev_c, test_c)  # 报告结果

    def save_checkpoints(self):
        """
        保存模型检查点。
        """
        cpt_path = os.path.join(self.cfg.output_dir, 'checkpoint')
        if not os.path.exists(cpt_path):
            os.makedirs(cpt_path)
        self.model.save_pretrained(cpt_path)  # 保存模型

    def eval_and_update(self, global_step):
        """
        评估并更新模型（需要子类实现）。
        """
        raise NotImplementedError()

    def report_result(self, dev_c, test_c, global_step=None):
        """
        报告评估结果（需要子类实现）。
        """
        raise NotImplementedError()


class Runner(BaseRunner):
    """
    运行器类，继承自 BaseRunner，提供具体实现。
    """
    def __init__(self, cfg=None, data_samples=None, data_features=None, data_loaders=None, model=None, optimizer=None, scheduler=None, metric_fn_dict=None):
        super().__init__(cfg, data_samples, data_features, data_loaders, model, optimizer, scheduler, metric_fn_dict)
        # 初始化评估指标
        self.metric = {
            "best_dev_f1": 0.0,  # 最佳验证集 F1 分数
            "related_test_f1": 0.0,  # 相关测试集 F1 分数
        }

        # 定义评估函数字典
        self.metric_fn_dict = {
            "span": eval_std_f1_score,  # 标准 F1 分数
            "text": eval_text_f1_score,  # 文本 F1 分数
            "head": eval_head_f1_score,  # 头部 F1 分数
        }
        # 更新评估器的评估函数字典
        self.dev_evaluator.metric_fn_dict = self.metric_fn_dict
        self.test_evaluator.metric_fn_dict = self.metric_fn_dict

    def eval_and_update(self, global_step):
        """
        评估并更新模型。
        """
        dev_c, _ = self.dev_evaluator.evaluate()  # 验证集评估
        test_c, _ = self.test_evaluator.evaluate()  # 测试集评估

        output_dir = os.path.join(self.cfg.output_dir, 'checkpoint')
        os.makedirs(output_dir, exist_ok=True)

        dev_f1, test_f1 = dev_c["f1"], test_c["f1"]
        if dev_f1 > self.metric["best_dev_f1"]:  # 如果验证集 F1 分数更优
            self.metric["best_dev_f1"] = dev_f1
            self.metric["related_test_f1"] = test_f1

            self.report_result(dev_c, test_c, global_step)  # 报告结果
            self.save_checkpoints()  # 保存检查点
        logger.info('current best dev-f1 score: {}'.format(self.metric["best_dev_f1"]))
        logger.info('current related test-f1 score: {}'.format(self.metric["related_test_f1"]))

    def report_result(self, dev_c, test_c, global_step=None):
        """
        报告评估结果。
        """
        show_results(self.test_features, os.path.join(self.cfg.output_dir, f'best_test_related_results.log'), 
            {"test related best score": f"P: {test_c['precision']} R: {test_c['recall']} f1: {test_c['f1']}", "global step": global_step}
        )
        show_results(self.dev_features, os.path.join(self.cfg.output_dir, f'best_dev_results.log'), 
            {"dev best score": f"P: {dev_c['precision']} R: {dev_c['recall']} f1: {dev_c['f1']}", "global step": global_step}
        )