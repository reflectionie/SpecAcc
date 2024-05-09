import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict
import itertools

def process_jsonl(file_name, categories):
    results = {}
    for category in categories:
        results[category] = {
            'accept_token_length': [],
            'time_elapsed': []
        }

    with open(file_name, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['category'] in categories:
                for eval_result in data['eval_results']:
                    results[data['category']]['accept_token_length'].extend(eval_result['accept_token_length'])
                    results[data['category']]['time_elapsed'].append(eval_result['time_elapsed'])

    return results

def draw_radar_chart(results_list, path, metric, file_name_list):
    categories = list(results_list[0].keys())
    num_vars = len(categories)

    # Compute angle of each axis in the plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Ensure that the plot is a complete circle
    angles += angles[:1]

    # Draw the plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Set category labels
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])  # cycle through colors defined in matplotlib

    for results, color in zip(results_list, colors):
        # Compute average of the chosen metric for each category
        avg_metric = [np.mean(results[cat][metric]) for cat in categories]

        # Ensure that the plot is a complete circle
        avg_metric += avg_metric[:1]

        ax.fill(angles, avg_metric, color=color, alpha=0.25)

    # Add legend
    ax.legend([f'Average {metric.capitalize()} for {file_name}' for file_name in file_name_list])

    # Save the plot
    plt.savefig(f'{path}/radar_chart_{metric}.png')

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--jsonl_paths', nargs='+', required=True, help="Paths to the answer jsonl files.")
parser.add_argument('--output_path', type=str, required=True, help="Path to save the output images.")
parser.add_argument('--file_name_list', type=str, nargs='+', required=True, help="File name list")
args = parser.parse_args()

# 使用函数
categories = ['writing', 'roleplay', 'reasoning','coding','extraction','stem','humanities','translation','summarization','qa','math_reasoning','rag']  # 你的类别列表
results_list = []

for jsonl_path in args.jsonl_paths:
    results_list.append(process_jsonl(jsonl_path, categories))

# Draw the radar chart for 'time_elapsed'
draw_radar_chart(results_list, args.output_path, 'time_elapsed', args.file_name_list)

# Draw the radar chart for 'accept_token_length'
draw_radar_chart(results_list, args.output_path, 'accept_token_length', args.file_name_list)

