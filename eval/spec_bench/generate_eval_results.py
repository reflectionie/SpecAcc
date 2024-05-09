from config.config_manager import Config

from generate.generate_input_ids import generate_input_ids


import torch
import argparse
# from fastchat.model import get_conversation_template
# from generate.generate_input_ids import generate_input_ids, generate_input_ids_attn_mask
# from generate.generate_output import generate_output,  generate_single_output_ids
from model.eagle.ea_model import EaModel
from model.load_model.load_speculative_model import load_speculative_model_eval





from eval.spec_bench.fastchat_answer import get_eval_results

if __name__ == "__main__":

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Your Program Description')
    parser.add_argument('--config', type=str,
                        help='Path to the config file', required=True)
    args = parser.parse_args()

    # 使用配置文件路径初始化配置
    config = Config.get_config(args.config)
    print(config)
    
    # fix random seed
    torch.manual_seed(config.get('seed', 0))

    # single output_ids
    speculative_model = load_speculative_model_eval(config)
    # if config["speculative_method"] in ["lookahead", "auto_regressive_greedy"]:
    #     input_ids_attn_mask = generate_input_ids_attn_mask(speculative_model, config)
    #     output = generate_single_output_ids(speculative_model, input_ids_attn_mask, config)
    # elif config["speculative_method"] == "eagle":
    #     input_ids = generate_input_ids(speculative_model, config)
    #     output = generate_single_output_ids(speculative_model, input_ids, config)
    # elif config["speculative_method"] == "medusa":
    #     input_ids = generate_input_ids(speculative_model, config)
    #     output = generate_single_output_ids(speculative_model, input_ids, config)
    # elif config["speculative_method"] == "hydra":
    #     input_ids = generate_input_ids(speculative_model, config)
    #     output = generate_single_output_ids(speculative_model, input_ids, config)

    mean_accept_tokens, total_time, tokens_per_second = get_eval_results(
        speculative_model=speculative_model,
        tokenizer=speculative_model.tokenizer if config['speculative_method'] != "ensemble_hydra" else speculative_model['ensemble_model'][0].tokenizer,
        # generate_input_ids = generate_input_ids,
        # generate_single_output_ids = generate_single_output_ids,
        answer_file_path=None,
        max_new_tokens = 1024,
        num_choices = None,
        config = config
    )
    
    print('finished....')
    
    
    
    
    
    
    
    

