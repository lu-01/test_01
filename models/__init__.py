import sys
sys.path.append("../")  # 添加上级目录到系统路径，方便模块导入
import copy
import logging
logger = logging.getLogger(__name__)  # 设置日志记录器

from transformers import BartConfig, BartTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup

from .paie import PAIE  # 导入 PAIE 模型
from .single_prompt import BartSingleArg  # 导入单提示模型
from utils import EXTERNAL_TOKENS  # 导入特殊标记
from processors.processor_multiarg import MultiargProcessor  # 导入多参数处理器

# 定义模型类型与对应的配置、模型类和分词器类
MODEL_CLASSES = {
    'paie': (BartConfig, PAIE, BartTokenizerFast),
    'base': (BartConfig, BartSingleArg, BartTokenizerFast)
}


def build_model(args, model_type):
    """
    构建模型、分词器、优化器和学习率调度器。

    Args:
        args: 配置参数对象。
        model_type: 模型类型（'paie' 或 'base'）。

    Returns:
        model: 构建的模型对象。
        tokenizer: 分词器对象。
        optimizer: 优化器对象（如果是推理模式，则为 None）。
        scheduler: 学习率调度器对象（如果是推理模式，则为 None）。
    """
    # 根据模型类型获取对应的配置类、模型类和分词器类
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

    # 加载模型配置
    if args.inference_only:
        config = config_class.from_pretrained(args.inference_model_path)  # 推理模式下加载预训练配置
    else:
        config = config_class.from_pretrained(args.model_name_or_path)  # 训练模式下加载预训练配置
    config.model_name_or_path = args.model_name_or_path
    config.device = args.device
    config.context_representation = args.context_representation

    # 设置模型的长度相关参数
    config.max_enc_seq_length = args.max_enc_seq_length
    config.max_dec_seq_length = args.max_dec_seq_length
    config.max_prompt_seq_length = args.max_prompt_seq_length
    config.max_span_length = args.max_span_length

    # 设置模型的其他配置
    config.bipartite = args.bipartite
    config.matching_method_train = args.matching_method_train

    # 初始化分词器
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, add_special_tokens=True)

    # 加载模型
    if args.inference_only:
        model = model_class.from_pretrained(args.inference_model_path, from_tf=bool('.ckpt' in args.inference_model_path), config=config)
    else:
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    # 添加触发器特殊标记和提示标记
    new_token_list = copy.deepcopy(EXTERNAL_TOKENS)  # 深拷贝特殊标记列表
    prompts = MultiargProcessor._read_prompt_group(args.prompt_path)  # 读取提示组
    for event_type, prompt in prompts.items():
        token_list = prompt.split()  # 将提示拆分为单词列表
        for token in token_list:
            # 如果是新的特殊标记，添加到列表中
            if token.startswith('<') and token.endswith('>') and token not in new_token_list:
                new_token_list.append(token)
    tokenizer.add_tokens(new_token_list)  # 将新标记添加到分词器中
    logger.info("Add tokens: {}".format(new_token_list))  # 记录添加的标记
    model.resize_token_embeddings(len(tokenizer))  # 调整模型的词嵌入大小以适应新标记

    # 如果是推理模式，不需要优化器和调度器
    if args.inference_only:
        optimizer, scheduler = None, None
    else:
        # 准备优化器和学习率调度器（线性预热和衰减）
        no_decay = ['bias', 'LayerNorm.weight']  # 不进行权重衰减的参数
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)  # 初始化 AdamW 优化器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.max_steps * args.warmup_steps,  # 预热步数
            num_training_steps=args.max_steps  # 总训练步数
        )

    return model, tokenizer, optimizer, scheduler  # 返回模型、分词器、优化器和调度器