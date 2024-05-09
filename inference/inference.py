import argparse
from config.config_manager import Config
# from fastchat.model import get_conversation_template
from generate.generate_input_ids import generate_input_ids, generate_input_ids_attn_mask
from generate.generate_output import generate_output
from model.eagle.ea_model import EaModel
from model.load_model.load_speculative_model import load_speculative_model_eval

if __name__ == "__main__":

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Your Program Description')
    parser.add_argument('--config', type=str,
                        help='Path to the config file', required=True)
    args = parser.parse_args()

    # 使用配置文件路径初始化配置
    config = Config.get_config(args.config)
    print(config)

    speculative_model = load_speculative_model_eval(config)
    if config["speculative_method"] in ["lookahead", "auto_regressive_greedy"]:
        input_ids_attn_mask = generate_input_ids_attn_mask(speculative_model, config)
        output = generate_output(speculative_model, input_ids_attn_mask, config)
    elif config["speculative_method"] == "eagle":
        input_ids = generate_input_ids(speculative_model, config)
        output = generate_output(speculative_model, input_ids, config)
    elif config["speculative_method"] == "medusa":
        input_ids = generate_input_ids(speculative_model, config)
        output = generate_output(speculative_model, input_ids, config)
    elif config["speculative_method"] == "hydra":
        input_ids = generate_input_ids(speculative_model, config)
        output = generate_output(speculative_model, input_ids, config)
    print(output)
