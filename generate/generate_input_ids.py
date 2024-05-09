import torch
from fastchat.model import get_conversation_template


def generate_input_ids(model, config):
    your_message = '' + config['conv']['user_message']
    
    # 根据基础模型配置加载prompt
    if config['base_model']['base_model_name'] == 'vicuna':
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    if config['base_model']['base_model_name'] == 'llama' or 'tinyllama':
        conv = get_conversation_template("vicuna")  # 似乎只有vicuna的，全部get这个吧
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    # 根据speculative模型配置做tokenize
    if config["speculative_method"] == "eagle":
        input_ids = model.tokenizer([prompt]).input_ids
        input_ids = torch.as_tensor(input_ids).cuda()
    elif config["speculative_method"] == "medusa":
        input_ids = model.tokenizer([prompt]).input_ids
        input_ids = torch.as_tensor(input_ids).cuda()
    elif config["speculative_method"] == "hydra":
        input_ids = model.tokenizer([prompt]).input_ids
        input_ids = torch.as_tensor(input_ids).cuda()
    return input_ids

def generate_input_ids_attn_mask(model, config):
    your_message = '' + config['conv']['user_message']
    
    # 根据基础模型配置加载prompt
    if config['base_model']['base_model_name'] == 'vicuna':
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    if config['base_model']['base_model_name'] == 'llama' or 'tinyllama':
        conv = get_conversation_template("vicuna")  # 似乎只有vicuna的，全部get这个吧
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    # 根据speculative模型配置做tokenize
    if config["speculative_method"] == "lookahead" or "auto_regressive_greedy":
        input_ids_attn_mask = model.tokenizer([prompt],  return_tensors='pt')
        input_ids_attn_mask = input_ids_attn_mask.to("cuda")
    
    return input_ids_attn_mask