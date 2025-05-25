import os
# 设置环境变量，强制使用 Intel 的 MKL 服务
os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
# 如果环境变量 DEBUG 被设置为 True，打印调试模式信息
if os.environ.get('DEBUG', False): 
    print('\033[92m'+'Running code in DEBUG mode'+'\033[0m')

from tabulate import tabulate  # 导入 tabulate 用于表格打印

# 导入必要的模块和函数
from models import build_model  # 构建模型
from processors import build_processor  # 构建数据处理器
from utils import set_seed  # 设置随机种子
from runner.runner import Runner  # 导入运行器类
from utils import logger  # 导入日志记录器


def run(args, model, processor, optimizer, scheduler):
    """
    执行训练或推理任务的主函数。
    """
    set_seed(args)  # 设置随机种子，确保结果可复现

    # 生成训练数据加载器
    logger.info("train dataloader generation")
    train_examples, train_features, train_dataloader, args.train_invalid_num = processor.generate_dataloader('train')

    # 生成验证数据加载器
    logger.info("dev dataloader generation")
    dev_examples, dev_features, dev_dataloader, args.dev_invalid_num = processor.generate_dataloader('dev')

    # 生成测试数据加载器
    logger.info("test dataloader generation")
    test_examples, test_features, test_dataloader, args.test_invalid_num = processor.generate_dataloader('test')

    # # 初始化运行器
    # runner = Runner(
    #     cfg=args,  # 配置参数
    #     data_samples=[train_examples, dev_examples, test_examples],  # 数据样本
    #     data_features=[train_features, dev_features, test_features],  # 数据特征
    #     data_loaders=[train_dataloader, dev_dataloader, test_dataloader],  # 数据加载器
    #     model=model,  # 模型
    #     optimizer=optimizer,  # 优化器
    #     scheduler=scheduler,  # 学习率调度器
    #     metric_fn_dict=None,  # 评估指标函数字典（此处为 None）
    # )
    # runner.run()  # 根据配置执行训练或推理任务


def main():
    """
    主函数，负责初始化配置、构建模型和处理器，并调用运行函数。
    """
    from config_parser import get_args_parser  # 导入配置解析器
    args, help_dict = get_args_parser()  # 获取命令行或配置文件参数

    logger.info("========== 训练前参数列表 ==========")
    data = [[help_dict.get(k, ""), k, v] for k, v in vars(args).items()]
    headers = ["帮助信息", "参数名", "参数值"]
    logger.info("\n" + tabulate(data, headers=headers, tablefmt="grid"))
    logger.info("========== 参数打印完毕 ==========")

    set_seed(args)  # 设置随机种子，确保结果可复现
    logger.info("Random seed set to %d", args.seed)  # 打印随机种子

    # 构建模型、分词器、优化器和学习率调度器
    logger.info("Start building model, tokenizer, optimizer, and scheduler")
    model, tokenizer, optimizer, scheduler = build_model(args, args.model_type)
    logger.info("Model, tokenizer, optimizer, and scheduler built successfully")
    model.to(args.device)  # 将模型移动到指定设备（如 GPU 或 CPU）
    logger.info("Model loaded to device %s", args.device)  # 打印模型加载设备信息

    # 构建数据处理器
    logger.info("开始构建数据处理器")
    processor = build_processor(args, tokenizer)
    logger.info("数据处理器构建完成")

    # 打印训练/评估参数
    logger.info("========== 训练后参数列表 ==========")
    data = [[help_dict.get(k, ""), k, v] for k, v in vars(args).items()]
    headers = ["帮助信息", "参数名", "参数值"]
    logger.info("\n" + tabulate(data, headers=headers, tablefmt="grid"))
    logger.info("========== 参数打印完毕 ==========")

    # 调用运行函数，执行训练或推理任务
    run(args, model, processor, optimizer, scheduler)


if __name__ == "__main__":
    # 如果脚本作为主程序运行，则调用 main 函数
    main()