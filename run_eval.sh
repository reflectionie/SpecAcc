#!/bin/bash

PYTHONPATH=. python ./eval/spec_bench/generate_eval_results.py --config ./config/config_files/2024_04_13/test_eagle_inference.yaml

PYTHONPATH=. python ./eval/spec_bench/generate_eval_results.py --config ./config/config_files/2024_04_13/test_hydra_inference.yaml

PYTHONPATH=. python ./eval/spec_bench/generate_eval_results.py --config ./config/config_files/2024_04_13/test_medusa_inference.yaml

PYTHONPATH=. python eval/spec_bench/result_analysis.py --answer ./eval/spec_bench/2024_04_13/medusa_answer.jsonl --output_path ./eval/spec_bench/2024_04_13/ --file_name medusa

PYTHONPATH=. python eval/spec_bench/result_analysis.py --answer ./eval/spec_bench/2024_04_13/eagle_answer.jsonl --output_path ./eval/spec_bench/2024_04_13/ --file_name eagle

PYTHONPATH=. python eval/spec_bench/result_analysis.py --answer ./eval/spec_bench/2024_04_13/hydra_answer.jsonl --output_path ./eval/spec_bench/2024_04_13/ --file_name hydra

PYTHONPATH=. python eval/spec_bench/result_analysis_all.py --jsonl_paths ./eval/spec_bench/2024_04_13/medusa_answer.jsonl ./eval/spec_bench/2024_04_13/hydra_answer.jsonl ./eval/spec_bench/2024_04_13/eagle_answer.jsonl --output_path ./eval/spec_bench/2024_04_13/ --file_name_list medusa hydra eagle