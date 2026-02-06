python3 src/context_compression/run.py \
    +experiments_longbench_narrativeqa=evaluate_llama_compress_zeroshot_qa_narrativeqa \
    custom_datasets=narrative_qa_custom \
    models.pretrained_model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct" \
    tokenizers.pretrained_model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct" \
    trainers.evaluation_config.output_dir="./results/llama3_narrativeqa"

# python3 src/context_compression/run.py \
#     +experiments_longbench_qasper=evaluate_llama_compress_zeroshot_qa_qasper \
#     custom_datasets=qasper_qa_custom \
#     models.pretrained_model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct" \
#     tokenizers.pretrained_model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct" \
#     trainers.evaluation_config.output_dir="./results/llama3_qasper"

# python3 src/context_compression/run.py \
#     +experiments_longbench_multifieldqa=evaluate_llama_compress_zeroshot_qa_multifield \
#     custom_datasets=multifield_qa_custom \
#     models.pretrained_model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct" \
#     tokenizers.pretrained_model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct" \
#     trainers.evaluation_config.output_dir="./results/llama3_multifieldqa"