srun -p INTERN2 --gres=gpu:8 --quotatype=reserved --ntasks=1 --ntasks-per-node=1 --cpus-per-task=5 python llava/eval/cli_judge_yes.py \
    --model-path checkpoints/llava-7b-Lightning-finetune-evaluation \
    --image-folder \
    ../autoeval_det/data/coco/val2014 \
    --answers-file \
    res/llava-llama-2-13b-chat-lightning-preview-answer.jsonl \
    --tag llava-7b-Lightning-finetune-evaluation
