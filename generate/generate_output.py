import re
from model import lade
import torch
from eval_tools.time_tools import Timer
from model.hydra.hydra_model import ensemble_hydra_generate_reflectio

def get_last_element_of_generator(generator):
    """
    抽取返回生成器的LLM的输出
    """
    last_element = None
    for element in generator:
        last_element = element
    return last_element


def re_match(text):
    """
    正则匹配LLM的输出
    """
    pattern = r'ASSISTANT:\s*(.*?)(?=<\/s><s>)'
    # 使用re.search找到第一个匹配的内容
    match = re.search(pattern, text)
    # 如果找到了匹配项，则提取匹配的组
    if match:
        extracted_text = match.group(1)
        return extracted_text
    else:
        print("No match found.")
        return text


def generate_output(model, input_info, config):
    if config["speculative_method"] == "eagle":
        input_ids = input_info
        with Timer() as t:
            output_ids = model.eagle_generate(
                input_ids,
                temperature=0.5,
                max_new_tokens=512)
        
        output = model.tokenizer.decode(
            output_ids[0][input_info['input_ids'].shape[1]:])
        # output = re_match(output)
    elif config["speculative_method"] == "medusa":
        input_ids = input_info
        with Timer() as t:
            output_generator = model.medusa_generate(
                input_ids,
                temperature=0.5,
                max_steps=512,
            )
        output = get_last_element_of_generator(output_generator)['text']
    elif config["speculative_method"] == "hydra":
        input_ids = input_info
        with Timer() as t:
            output_generator = model.hydra_generate(
                input_ids,
                temperature=0.5,
                max_steps=512,
            )
        output = get_last_element_of_generator(output_generator)['text']
    elif config["speculative_method"] == 'lookahead':
        with Timer() as t:
            output_ids = model.generate(**input_info,
                                        max_new_tokens=256,
                                        do_sample=True,
                                        temperature=0.7,
                                        top_k=50, top_p=0.9)
        output = model.tokenizer.decode(
            output_ids[0], skip_special_tokens=False)
        # output = re_match(output)
    elif config["speculative_method"] == 'auto_regressive_greedy':
        with Timer() as t:
            output_ids = model.generate(input_info,
                                        max_new_tokens=256,
                                        do_sample=False)
        output = model.tokenizer.decode(
            output_ids[0], skip_special_tokens=False)
        # output = re_match(output)
    print(t.time_elapsed)
    return output


def generate_single_output_ids(model, input_info, config):
    """
    Parameters:
    model (model): spectulive model
    input_info (tensor or dict): for medusa, eagle, hydra: tensor (input_ids), for lookaheead: dict (input_ids and mask)

    Returns:
    output_ids (tensor) 
    t.time_elapsed (float): s
    accept_token_length (list): accepted token length
    """
    if config["speculative_method"] == "eagle":
        input_ids = input_info
        torch.cuda.synchronize()
        with Timer() as t:
            output_ids, accept_token_length = model.eagle_generate_reflectio(
                input_ids,
                temperature=0,
                max_new_tokens=512)  
        torch.cuda.synchronize()
        output_ids = output_ids[0][input_info.shape[1]:]
    elif config["speculative_method"] == "medusa":
        input_ids = input_info
        torch.cuda.synchronize()
        with Timer() as t:
            output_ids, accept_token_length = model.medusa_generate_reflectio(
                input_ids,
                temperature=0,
                max_new_tokens=512,
            )
        torch.cuda.synchronize()
        output_ids = output_ids[0][input_info.shape[1]:]
    elif config["speculative_method"] == "hydra":
        input_ids = input_info
        torch.cuda.synchronize()
        with Timer() as t:
            output_ids, accept_token_length = model.hydra_generate_reflectio(
                input_ids,
                temperature=0,
                max_new_tokens=512,
            )
        torch.cuda.synchronize()
        output_ids = output_ids[0][input_info.shape[1]:]
    elif config["speculative_method"] == 'lookahead':
        torch.cuda.synchronize()
        with Timer() as t:
            output_ids = model.generate(**input_info,
                                        max_new_tokens=256,
                                        do_sample=True,
                                        temperature=0.7,
                                        top_k=50, top_p=0.9)
        torch.cuda.synchronize()
        output_ids = output_ids[0][input_info.shape[1]:]
    elif config["speculative_method"] == 'auto_regressive_greedy':
        torch.cuda.synchronize()
        with Timer() as t:
            output_ids = model.generate(input_info,
                                        max_new_tokens=256,
                                        do_sample=False)
        torch.cuda.synchronize()
        output_ids = output_ids[0][input_info.shape[1]:]
    elif config["speculative_method"] == 'ensemble_hydra':
        input_ids = input_info
        torch.cuda.synchronize()
        with Timer() as t:
            output_ids, accept_token_length = ensemble_hydra_generate_reflectio(
                model,
                input_ids,
                temperature=0,
                max_new_tokens=512,
            )
        torch.cuda.synchronize()
        output_ids = output_ids[0][input_info.shape[1]:]
    
    # print(t.time_elapsed)
    return output_ids, t.time_elapsed, accept_token_length
