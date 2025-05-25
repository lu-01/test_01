# 检查脚本参数是否为空
if [ $# == 0 ] 
then
    # 如果没有传入参数，使用默认值
    SEED=42  # 随机种子
    LR=2e-5  # 学习率
else
    # 如果传入参数，使用用户提供的值
    SEED=$1  # 第一个参数作为随机种子
    LR=$2  # 第二个参数作为学习率
fi

# 定义工作路径，包含随机种子和学习率的目录结构
work_path=exps/wikievent/$SEED/$LR
mkdir -p $work_path  # 创建工作目录（如果不存在）

# 执行训练脚本
python -u engine.py \
    --model_type=paie \
    --dataset_type=wikievent \
    --model_name_or_path=D:/huhaiyang/huggingface_models/bart-base \
    --role_path=./data/dset_meta/description_wikievent.csv \
    --prompt_path=./data/prompts/prompts_wikievent_full.csv \
    --seed=$SEED \
    --output_dir=$work_path \
    --learning_rate=$LR \
    --max_steps=10000 \
    --max_enc_seq_length 500 \
    --max_prompt_seq_length 80 \
    --bipartite