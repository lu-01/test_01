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
import logging  # 导入日志模块
logger = logging.getLogger(__name__)  # 创建日志记录器，使用当前模块的名称

def run(args, model, processor, optimizer, scheduler):
    """
    执行训练或推理任务的主函数。
    Args:
        args: 配置参数对象，包含训练或推理的相关设置。
        model: 构建好的模型对象。
        processor: 数据处理器对象，用于加载和处理数据。
        optimizer: 优化器对象，用于模型参数更新。
        scheduler: 学习率调度器对象，用于调整学习率。
    """
    logger.info("========== 进入run函数 ==========")  # 打印开始执行引擎的日志

    set_seed(args)  # 设置随机种子，确保结果可复现
    logger.info("Random seed set to %d", args.seed)  # 打印随机种子

    # 生成训练数据加载器
    logger.info("train dataloader generation")
    train_examples, train_features, train_dataloader, args.train_invalid_num = processor.generate_dataloader('train')

    # 生成验证数据加载器
    logger.info("dev dataloader generation")
    dev_examples, dev_features, dev_dataloader, args.dev_invalid_num = processor.generate_dataloader('dev')

    # 生成测试数据加载器
    logger.info("test dataloader generation")
    test_examples, test_features, test_dataloader, args.test_invalid_num = processor.generate_dataloader('test')

    # 初始化运行器
    runner = Runner(
        cfg=args,  # 配置参数
        data_samples=[train_examples, dev_examples, test_examples],  # 数据样本
        data_features=[train_features, dev_features, test_features],  # 数据特征
        data_loaders=[train_dataloader, dev_dataloader, test_dataloader],  # 数据加载器
        model=model,  # 模型
        optimizer=optimizer,  # 优化器
        scheduler=scheduler,  # 学习率调度器
        metric_fn_dict=None,  # 评估指标函数字典（此处为 None）
    )
    runner.run()  # 根据配置执行训练或推理任务


def main():
    """
    主函数，负责初始化配置、构建模型和处理器，并调用运行函数。
    """
    logger.info("========== 进入main函数 ==========")  # 打印开始执行引擎的日志

    from config_parser import get_args_parser  # 导入配置解析器
    args, help_dict = get_args_parser()  # 获取命令行或配置文件参数
    logger.info("========== 参数解析完成 ==========")  # 打印参数解析完成的日志

    # 设置日志 如果不是推理模式
    if not args.inference_only:
        logger.info(f"Output full path {os.path.join(os.getcwd(), args.output_dir)}")
        # 如果输出目录不存在，则创建
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        # 配置日志：输出到文件和控制台
        logging.basicConfig(
            filename=os.path.join(args.output_dir, "log.txt"),  # 日志文件路径
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',  # 日志格式
            datefmt='%m/%d/%Y %H:%M:%S',  # 日期格式
            level=logging.INFO,  # 日志级别
            encoding='utf-8',    # 指定文件编码（Python 3.9+ 支持）
        )
    else:  # 如果是推理模式
        # 配置日志：仅输出到控制台
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',  # 日志格式
            datefmt='%m/%d/%Y %H:%M:%S',  # 日期格式
            level=logging.INFO,  # 日志级别
            encoding='utf-8',    # 指定文件编码（Python 3.9+ 支持）
        )
    logger.info("========== 日志模式设置完成 ==========")  # 打印日志设置完成的日志

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
    logger.info("========== 数据处理完成后的参数列表 ==========")
    data = [[help_dict.get(k, ""), k, v] for k, v in vars(args).items()]
    headers = ["帮助信息", "参数名", "参数值"]
    logger.info("\n" + tabulate(data, headers=headers, tablefmt="grid"))
    logger.info("========== 参数打印完毕 ==========")

    # 调用运行函数，执行训练或推理任务
    run(args, model, processor, optimizer, scheduler)

    logger.info("========== 训练/推理任务完成 ==========")  # 打印任务完成日志
    logger.info("请查看输出目录 %s 中的日志和结果文件", args.output_dir)  # 提示用户查看输出目录
    logger.info("========== 退出main函数 ==========")  # 打印退出主函数的日志


if __name__ == "__main__":
    # 如果脚本作为主程序运行，则调用 main 函数
    main()