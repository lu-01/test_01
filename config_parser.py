import argparse


def get_args_parser():
    """
    定义并解析命令行参数或配置文件参数。
    """
    parser = argparse.ArgumentParser(add_help=False)

    # 模型相关参数
    parser.add_argument("--model_type", default='paie', choices=["paie", "base"], type=str,
                        help="选择模型类型：多提示模型(paie)或单提示模型(base)")
    parser.add_argument("--model_name_or_path", default="./ckpts/bart-base", type=str,
                        help="预训练语言模型的路径")
    parser.add_argument("--dataset_type", default="rams", choices=["ace_eeqa", "rams", "wikievent"], type=str,
                        help="数据集类型：句子级(ace_eeqa)或文档级(rams/wikievent)")
    parser.add_argument("--role_path", default='./data/dset_meta/description_rams.csv', type=str,
                        help="包含所有角色名称的文件路径，用于访问数据集的参数角色")
    parser.add_argument("--prompt_path", default='./data/prompts/prompts_rams_full.csv', type=str,
                        help="包含所有提示的文件路径，用于数据集")
    parser.add_argument("--output_dir", default='./outputs_res', type=str,
                        help="输出目录，用于存储检查点和日志文件")
    parser.add_argument("--keep_ratio", default=1.0, type=float,
                        help="保留训练样本的比例，用于小样本设置")
    parser.add_argument('--inference_only', default=False, action="store_true",
                        help="如果设置为 True，模型将直接进行推理而不进行训练")

    # 数据处理相关参数
    parser.add_argument("--pad_mask_token", default=0, type=int,
                        help="填充标记的 ID")
    parser.add_argument('--logging_steps', default=100, type=int,
                        help="记录日志的步数间隔")
    parser.add_argument('--eval_steps', default=500, type=int,
                        help="验证的步数间隔")
    parser.add_argument("--max_span_length", default=10, type=int,
                        help="提取参数的最大跨度长度（启发式约束）")
    parser.add_argument("--batch_size", default=4, type=int, 
                        help="训练时的批次大小（包含反向传播）")
    parser.add_argument("--infer_batch_size", default=32, type=int, 
                        help="推理时的批次大小（不包含反向传播）")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                        help="在执行一次反向传播/更新之前累积的更新步数")
    parser.add_argument("--max_enc_seq_length", default=500, type=int,
                        help="上下文的最大长度")
    parser.add_argument("--window_size", default=250, type=int,
                        help="对于超出长度限制的文档，在触发词周围添加窗口并丢弃窗口外的单词")
    parser.add_argument('--context_representation', default="decoder", choices=['encoder', 'decoder'], type=str,
                        help="使用完整的 BART（decoder）还是仅使用 BART-encoder（encoder）表示上下文")

    # 优化器相关参数
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="权重衰减系数")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Adam 优化器的 epsilon 值")
    parser.add_argument("--max_grad_norm", default=5.0, type=float,
                        help="梯度裁剪的最大范数")
    parser.add_argument("--max_steps", default=10000, type=int,
                        help="最大训练步数")
    parser.add_argument("--warmup_steps", default=0.1, type=float,
                        help="学习率预热步数的比例")
    parser.add_argument('--seed', default=42, type=int,
                        help="随机种子")
    parser.add_argument("--device", default='cuda', type=str,
                        help="运行设备（如 'cuda' 或 'cpu'）")

    # 推理模式相关参数
    parser.add_argument('--inference_model_path', default="./exps/rams_exp_0306_3/42/2e-5/checkpoint", type=str,
                        help="用于推理的检查点路径")

    # 单提示模型（base model）相关参数
    parser.add_argument("--max_dec_seq_length", default=20, type=int,
                        help="单提示的最大长度")
    parser.add_argument("--max_span_num", default=1, type=int,
                        help="每个角色提取的最大参数数量")
    parser.add_argument('--th_delta', default=.0, type=float,
                        help="控制是否接受候选跨度作为参数的阈值")

    # 多提示模型（paie model）相关参数
    parser.add_argument("--max_prompt_seq_length", default=64, type=int,
                        help="多提示的最大长度")
    parser.add_argument('--matching_method_train', default="max", choices=["max", 'accurate'], type=str,
                        help="训练期间的起始/结束标记匹配方法")
    parser.add_argument('--bipartite', default=False, action="store_true",
                        help="训练期间是否使用双向匹配损失")

    # 解析参数
    args = parser.parse_args()

    # 如果是推理模式，设置输出目录为推理模型路径的父目录
    if args.inference_only:
        args.output_dir = "/".join(args.inference_model_path.split("/")[:-1])

    help_dict = {}
    for action in parser._actions:
        if action.dest != 'help':
            help_dict[action.dest] = action.help
    return args, help_dict
