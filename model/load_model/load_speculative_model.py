from model.eagle.ea_model import EaModel
from model.medusa.medusa_model import MedusaModel
from model.hydra.hydra_model import HydraModel, HydraModelEnsembleHead
from model.hydra.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from model import lade


def load_speculative_model_eval(config):
    speculative_method = config['speculative_method']
    speculative_model_path = config['speculative_model']['speculative_model_path']
    speculative_model_name = config['speculative_model']['speculative_model_name']
    if speculative_model_name == 'eagle':
        speculative_model = EaModel.from_pretrained(
            base_model_path=config['base_model']['base_model_path'],
            ea_model_path=speculative_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    elif speculative_model_name == 'medusa':
        speculative_model = MedusaModel.from_pretrained(
            config['speculative_model']['speculative_model_path'],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_8bit=config['speculative_model']['load_in_8bit'],
            load_in_4bit=config['speculative_model']['load_in_4bit'],
        )
        
    elif speculative_model_name == 'hydra':
        speculative_model = HydraModel.from_pretrained(
            hydra_head_name_or_path=config['speculative_model']['speculative_model_path'],
            base_model=config['base_model']['base_model_path'],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_8bit=config['speculative_model']['load_in_8bit'],
            load_in_4bit=config['speculative_model']['load_in_4bit'],
        )
    elif speculative_model_name == 'tinyllama' and speculative_method == 'lookahead':
        lade.augment_all()
        # For a 7B model, set LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7
        # lade.config_lade(LEVEL=7, WINDOW_SIZE=20, GUESS_SET_SIZE=20, DEBUG=1, POOL_FROM_PROMPT=True)
        lade.config_lade(LEVEL=4, WINDOW_SIZE=5, GUESS_SET_SIZE=5,
                         DEBUG=1, POOL_FROM_PROMPT=True)
        speculative_model = AutoModelForCausalLM.from_pretrained(
            config['speculative_model']['speculative_model_path'],
            torch_dtype=torch.float16,
            device_map='cuda')
        speculative_model.tokenizer = AutoTokenizer.from_pretrained(
            config['speculative_model']['speculative_model_path'])
    elif speculative_model_name == 'ensemble_hydra':
        base_model =  KVLlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config['base_model']['base_model_path'],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_8bit=config['speculative_model']['load_in_8bit'],
            load_in_4bit=config['speculative_model']['load_in_4bit'],
        )
        ensemble_model = []
        for model_path in config['speculative_model']['speculative_model_path']:
            speculative_model = HydraModelEnsembleHead.from_pretrained(
                hydra_head_name_or_path=model_path,
                base_model=base_model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                load_in_8bit=config['speculative_model']['load_in_8bit'],
                load_in_4bit=config['speculative_model']['load_in_4bit'],
            )
            ensemble_model.append(speculative_model)
            
            
            
            
    if speculative_model_name == 'ensemble_hydra':
        for speculative_model in ensemble_model:
            speculative_model.eval()
        return {'base_model': base_model, 'ensemble_model': ensemble_model}
    else:
        speculative_model.eval()
        return speculative_model
