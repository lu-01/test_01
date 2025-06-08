import json

from .processor_multiarg import MultiargProcessor  # 导入多参数处理器类
import logging
logger = logging.getLogger(__name__)  # 设置日志记录器

# 定义数据集目录和相关文件路径
_DATASET_DIR = {
    'ace_eeqa': {
        "train_file": './data/ace_eeqa/train_convert.json',  # ACE 数据集的训练文件路径
        "dev_file": './data/ace_eeqa/dev_convert.json',  # ACE 数据集的验证文件路径
        "test_file": './data/ace_eeqa/test_convert.json',  # ACE 数据集的测试文件路径
        "max_span_num_file": "./data/dset_meta/role_num_ace.json",  # ACE 数据集的最大跨度数量文件路径
    },
    'rams': {
        "train_file": './data/RAMS_1.0/data/train.jsonlines',  # RAMS 数据集的训练文件路径
        "dev_file": './data/RAMS_1.0/data/dev.jsonlines',  # RAMS 数据集的验证文件路径
        "test_file": './data/RAMS_1.0/data/test.jsonlines',  # RAMS 数据集的测试文件路径
        "max_span_num_file": "./data/dset_meta/role_num_rams.json",  # RAMS 数据集的最大跨度数量文件路径
    },
    "wikievent": {
        "train_file": './data/WikiEvent/data/train.jsonl',  # WikiEvent 数据集的训练文件路径
        "dev_file": './data/WikiEvent/data/dev.jsonl',  # WikiEvent 数据集的验证文件路径
        "test_file": './data/WikiEvent/data/test.jsonl',  # WikiEvent 数据集的测试文件路径
        "max_span_num_file": "./data/dset_meta/role_num_wikievent.json",  # WikiEvent 数据集的最大跨度数量文件路径
    },
}


def build_processor(args, tokenizer):
    """
    构建数据处理器。

    Args:
        args: 配置参数对象，包含数据集类型和模型类型等信息。
        tokenizer: 分词器对象，用于处理文本。

    Returns:
        processor: 数据处理器对象，用于加载和处理数据。
    """
    logger.info("========== 进入build_processor函数 ==========")  # 打印开始执行数据处理器构建的日志

    logger.info(f"当前模型类型: {args.model_type}")  # 打印当前模型类型
    logger.info("开始检查数据集类型是否有效")
    # 检查数据集类型是否有效
    if args.dataset_type not in _DATASET_DIR: 
        logger.error(f"无效的数据集类型: {args.dataset_type}")
        raise NotImplementedError("Please use valid dataset name")  # 如果数据集类型无效，抛出异常
    logger.info("数据集类型检查通过" + f"当前数据集类型: {args.dataset_type}")  # 打印数据集类型检查通过的日志

    # 根据数据集类型设置训练、验证和测试文件路径
    logger.info(f"设置数据集 {args.dataset_type} 的文件路径")
    args.train_file = _DATASET_DIR[args.dataset_type]['train_file']
    args.dev_file = _DATASET_DIR[args.dataset_type]['dev_file']
    args.test_file = _DATASET_DIR[args.dataset_type]['test_file']
    logger.info(f"已设置数据集 {args.dataset_type} 的文件路径")

    # 如果模型类型为 "base"，加载最大跨度数量字典
    logger.info(f"当前模型类型: {args.model_type}\t" + "如果模型类型为 'base'，加载最大跨度数量字典")
    if args.model_type == "base":
        logger.info(f"加载最大跨度数量字典: {_DATASET_DIR[args.dataset_type]['max_span_num_file']}")
        with open(_DATASET_DIR[args.dataset_type]['max_span_num_file']) as f:
            args.max_span_num_dict = json.load(f)  # 加载 JSON 文件，存储为字典
        logger.info("最大跨度数量字典加载完成")

    # 初始化多参数处理器，读取参数角色参数模板，定义其他数据处理操作
    logger.info("初始化 MultiargProcessor")
    processor = MultiargProcessor(args, tokenizer)
    logger.info("数据处理器构建完成")

    logger.info("========== 退出build_processor函数 ==========")  # 打印结束执行数据处理器构建的日志
    return processor  # 返回数据处理器对象