import torch
from generate.generate_input_ids import generate_input_ids, generate_input_ids_attn_mask
from generate.generate_output import generate_output,  generate_single_output_ids

# fast chat
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm
import shortuuid
import time

import os
import json  

import statistics

# adapted from https://github1s.com/hemingkx/Spec-Bench/blob/main/evaluation/eval.py

@torch.inference_mode()
def get_eval_results(
        speculative_model,
        tokenizer, 
        # generate_input_ids,
        # generate_single_output_ids,
        # model_id,
        # questions,
        answer_file_path,
        max_new_tokens,
        num_choices,
        config,
        **kwargs,
):
    mean_accept_tokens = 0
    total_time = 0
    tokens_per_second = 0
    
    questions = load_questions(config['question_file'], config['question_begin'], config['question_end'])
    total_accept_length = []
    total_time_elapsed = []
    
    
    for question in tqdm(questions):
        conv = get_conversation_template("vicuna")
        eval_results = []
        for turn in range(len(question["turns"])):
            turn_question = question["turns"][turn]
            conv.append_message(conv.roles[0], turn_question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            # inputs = tokenizer([prompt], return_tensors="pt")
            input_ids = inputs.input_ids
            output_ids, time_elapsed, accept_token_length = generate_single_output_ids(
                model=speculative_model,
                input_info=input_ids,
                config=config
            )
            
            total_accept_length.append(accept_token_length)
            
            output = tokenizer.decode(
                output_ids
            )
            
            # print(output)
            # turns.append(output)
            total_time_elapsed.append(time_elapsed)
            conv.messages[-1][-1] = output
            eval_json = {
                'turn': turn,
                'time_elapsed': time_elapsed,
                'accept_token_length': accept_token_length,
                'model_output': output
            }
            
            eval_results.append(eval_json)
        
        # Dump answers
        os.makedirs(os.path.dirname(config['answer_file']), exist_ok=True)
        with open(os.path.expanduser(config['answer_file']), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": config['speculative_method'],
                "tstamp": time.time(),
                "eval_results": eval_results
            }
            fout.write(json.dumps(ans_json) + "\n")
        
    forward_mean_accept_tokens = statistics.mean([statistics.mean(accept_length) for accept_length in total_accept_length])
    total_time = sum(time_elapsed for time_elapsed in total_time_elapsed)
    tokens_per_second = sum([sum(accept_length) for accept_length in total_accept_length])/total_time
      
    return forward_mean_accept_tokens, total_time, tokens_per_second
    
