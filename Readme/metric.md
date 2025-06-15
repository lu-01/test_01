# def eval_rpf
```py
# 计算召回率、精确率和 F1 分数
def eval_rpf(gt_num, pred_num, correct_num)
    """
    计算召回率、精确率和 F1 分数。
    Args:
        gt_num: 真实值数量。
        pred_num: 预测值数量。
        correct_num: 正确匹配的数量。
    Returns:
        res: 包含召回率、精确率、F1 分数和数量统计的字典。
    """
    res = {
        "recall": recall, "precision": precision, "f1": f1,
        "gt_num": gt_num, "pred_num": pred_num, "correct_num": correct_num,
    }
    return res
```
# def eval_std_f1_score
```py
def eval_std_f1_score(features, invalid_gt_num=0):
    """
    计算标准 F1 分数。
    Args:
        features: 特征列表。
        invalid_gt_num: 无效的真实值数量。
    Returns:
        res_classification: 分类任务的评估结果。  识别角色且预测跨度 的召回率、精确率、F1 分数和数量统计的字典。
        res_identification: 识别任务的评估结果。  预测跨度 的召回率、精确率、F1 分数和数量统计的字典。
    """
    features = [feature, ...]
    features = InputFeatures(object):
        example_id: 样本文档名。
        feature_id: 特征 ID。
        event_type: 事件类型。
        event_trigger: 事件触发器信息。
        enc_text: 编码器输入文本。
        enc_input_ids: 编码器 输入。
        enc_mask_ids: 编码器 掩码。
        dec_prompt_text: 解码器  提示文本。
        dec_prompt_ids: 解码器 提示 ID。
        dec_prompt_mask_ids: 解码器 提示掩码 ID。
        arg_quries: 参数查询信息。
        arg_joint_prompt: 参数联合提示信息。（提示模板中角色信息）
        target_info: 目标信息。（实际事件的角色信息）
        old_tok_to_new_tok_index: 旧到新 token 的索引映射。
        full_text: 事件文本的完整内容。['Roadside', 'IED', 'kills', 'Russian', 'major', 'general', 'in', 'Syria', 'The', 'desert', 'south', 'and', 'east', 'of', 'Deir', 'ez', '-', 'Zor', 'on', 'the', 'regime', '’', 's', 'side', 'of', 'the', 'Euphrates', 'is', 'rife', 'with', 'insurgents', 'Russia', '’', 's', 'foray', 'into', 'Syria', 'took', 'yet', 'another', 'heavy', 'toll', 'this', 'week', ',', 'when', 'that', 'nation', '’', 's', 'Defense', 'Ministry', 'confirmed', 'that', 'a', 'major', 'general', 'was', 'killed', ...]
        arg_list: 参数列表。（当前事件类型的所有参数角色） ['Victim', 'Place', 'Killer', 'MedicalIssue']

        推测后的新增属性：
        gt_dict_word：{'Victim': [(5，5)], 'Place': [(7，7)]}
        pred_dict_word：{'Victim': [(5，5)], 'Place': [(7，7)]}
    res_classification = eval_rpf(gt_num + invalid_gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify + invalid_gt_num, pred_num_identify, correct_identify_num)

    return res_classification, res_identification
```
# def eval_text_f1_score
```py
    """
    计算基于文本的 F1 分数。
    Args:
        features: 特征列表。
        invalid_gt_num: 无效的真实值数量。
    Returns:
        res_classification: 分类任务的评估结果。识别角色且预测文本 的召回率、精确率、F1 分数和数量统计的字典。
        res_identification: 识别任务的评估结果。预测文本 的召回率、精确率、F1 分数和数量统计的字典。
    """
    res_classification = eval_rpf(gt_num + invalid_gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify + invalid_gt_num, pred_num_identify, correct_identify_num)
    return res_classification, res_identification
```
# eval_head_f1_score
```py
    """
    计算基于头部的 F1 分数。仅匹配论元的头部词（Head Word）​​，而不是整个论元跨度（Span），从而减少对边界错误的敏感性

    Args:
        features: 特征列表。
        invalid_gt_num: 无效的真实值数量。

    Returns:
        res_classification: 分类任务的评估结果。
        res_identification: 识别任务的评估结果。
    """
    res_classification = eval_rpf(gt_num + invalid_gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify + invalid_gt_num, pred_num_identify, correct_identify_num)
    return res_classification, res_identification
```
# show_results
```py
    """
    用于 ​​可视化评估结果​​ 并 ​​将详细匹配情况写入文件​​
    适用于 ​​事件抽取（Event Extraction）​​ 或 ​​论元角色分类（Argument Role Classification）​​ 任务

    Args:
        features: 特征列表。
        output_file: 输出文件路径。
        metainfo: 元信息字典。
    """
    dev best score: P: 0.5866666666666667 R: 0.616822429906542 f1: 0.6013667425968108
    global step: 7499
    -------------------------------------------------------------------------------------
    Sent: Roadside IED <t> kills </t> Russian major general in Syria The desert south and east of Deir ez - Zor on the regime ’ s side of the Euphrates is rife with insurgents Russia ’ s foray into Syria took yet another heavy toll this week , when that nation ’ s Defense Ministry confirmed that a major general was killed in Syria by an improvised explosive device , Al - Monitor online reported . Major General Vyacheslav Gladkikh died after a roadside IED detonated under a convoy of Russian soldiers and Syrian pro - regime militiamen near the city of Deir ez - Zor . Three other Russian military personnel were wounded , Russia ’ s state - run Tass news agency reported . A local commander of Syria ’ s National Defense Forces , a pro - Assad militia , was also reportedly killed . “ They ’ ve changed their tactics there , ” said one source , who declined to be named for security reasons . “ They do nighttime infiltrations , lay mines and booby traps . ” Video purporting to show the explosion circulated on social media this week . It was the first reported death of a Russian general in Syria since 2017 , when a lieutenant general was killed in the same province , reportedly by mortar fire from the Islamic State , Al - Monitor reported . Deir ez - Zor and nearly all of Syrian territory west of the Euphrates River lie in
    Event type: Life.Die.Unspecified			Trigger word: ['kills', [2, 3], 0]
    Example ID road_ied_8
    Arg Victim matched: Pred: general (5,5)	Gt: general (5,5)
    Arg Place matched: Pred: Syria (7,7)	Gt: Syria (7,7)
    -------------------------------------------------------------------------------------
    Sent: Roadside IED kills Russian major general in Syria The desert south and east of Deir ez - Zor on the regime ’ s side of the Euphrates is rife with insurgents Russia ’ s foray into Syria took yet another heavy toll this week , when that nation ’ s Defense Ministry confirmed that a major general was <t> killed </t> in Syria by an improvised explosive device , Al - Monitor online reported . Major General Vyacheslav Gladkikh died after a roadside IED detonated under a convoy of Russian soldiers and Syrian pro - regime militiamen near the city of Deir ez - Zor . Three other Russian military personnel were wounded , Russia ’ s state - run Tass news agency reported . A local commander of Syria ’ s National Defense Forces , a pro - Assad militia , was also reportedly killed . “ They ’ ve changed their tactics there , ” said one source , who declined to be named for security reasons . “ They do nighttime infiltrations , lay mines and booby traps . ” Video purporting to show the explosion circulated on social media this week . It was the first reported death of a Russian general in Syria since 2017 , when a lieutenant general was killed in the same province , reportedly by mortar fire from the Islamic State , Al - Monitor reported . Deir ez - Zor and nearly all of Syrian territory west of the Euphrates River lie in
    Event type: Life.Die.Unspecified			Trigger word: ['killed', [58, 59], 0]
    Example ID road_ied_8
    Arg Victim matched: Pred: general (56,56)	Gt: general (56,56)
    Arg Place dismatched: Pred: Syria (60,60)	Gt: __ No answer __ (-1,-1)
    -------------------------------------------------------------------------------------
    Sent: Roadside IED kills Russian major general in Syria The desert south and east of Deir ez - Zor on the regime ’ s side of the Euphrates is rife with insurgents Russia ’ s foray into Syria took yet another heavy toll this week , when that nation ’ s Defense Ministry confirmed that a major general was killed in Syria by an improvised explosive device , Al - Monitor online <t> reported </t> . Major General Vyacheslav Gladkikh died after a roadside IED detonated under a convoy of Russian soldiers and Syrian pro - regime militiamen near the city of Deir ez - Zor . Three other Russian military personnel were wounded , Russia ’ s state - run Tass news agency reported . A local commander of Syria ’ s National Defense Forces , a pro - Assad militia , was also reportedly killed . “ They ’ ve changed their tactics there , ” said one source , who declined to be named for security reasons . “ They do nighttime infiltrations , lay mines and booby traps . ” Video purporting to show the explosion circulated on social media this week . It was the first reported death of a Russian general in Syria since 2017 , when a lieutenant general was killed in the same province , reportedly by mortar fire from the Islamic State , Al - Monitor reported . Deir ez - Zor and nearly all of Syrian territory west of the Euphrates River lie in
    Event type: Contact.Contact.Broadcast			Trigger word: ['reported', [71, 72], 0]
    Example ID road_ied_8
    Arg Communicator matched: Pred: Al - Monitor (67,69)	Gt: Al - Monitor (67,69)
    -------------------------------------------------------------------------------------
```
