import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

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

def draw_radar_chart(results, path, file_name, metric):
    categories = list(results.keys())
    num_vars = len(categories)

    # Compute average of the chosen metric for each category
    avg_metric = [np.mean(results[cat][metric]) for cat in categories]

    # Compute angle of each axis in the plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Ensure that the plot is a complete circle
    avg_metric += avg_metric[:1]
    angles += angles[:1]

    # Draw the plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, avg_metric, color='blue', alpha=0.25)

    # Set category labels
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Add legend
    ax.legend([f'Average {metric.capitalize()}'])

    # Save the plot
    plt.savefig(f'{path}/radar_chart_{file_name}_{metric}.png')

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--answer', type=str, required=True, help="Path to the answer jsonl file.")
parser.add_argument('--output_path', type=str, required=True, help="Path to save the output images.")
parser.add_argument('--file_name', type=str, required=True, help="File name prefix.")
args = parser.parse_args()

# 使用函数
categories = ['writing', 'roleplay', 'reasoning','coding','extraction','stem','humanities','translation','summarization','qa','math_reasoning','rag']  # 你的类别列表
results = process_jsonl(args.answer, categories)

# Draw the radar chart for 'time_elapsed'
draw_radar_chart(results, args.output_path, args.file_name, 'time_elapsed')

# Draw the radar chart for 'accept_token_length'
draw_radar_chart(results, args.output_path, args.file_name, 'accept_token_length')
