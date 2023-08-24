srun -p INTERN2 --gres=gpu:8 --quotatype=reserved --ntasks=1 --ntasks-per-node=1 --cpus-per-task=5 python llava/eval/cli_judge_yes.py \
    --model-path liuhaotian/llava-llama-2-13b-chat-lightning-preview  #liuhaotian/LLaVA-Lightning-MPT-7B-preview \
    # --question-file \
    # playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    ../autoeval_det/data/coco/val2014 \
    --answers-file \
    res/llava-llama-2-13b-chat-lightning-preview-answer.jsonl \
    --tag \
    llava-llama-2-13b-chat-lightning-preview
