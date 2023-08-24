srun -p INTERN2 --gres=gpu:8 --quotatype=reserved --ntasks=1 --ntasks-per-node=1 --cpus-per-task=5 python llava/eval/model_vqa_detail.py \
    --model-path liuhaotian/LLaVA-Lightning-MPT-7B-preview \
    --question-file \
    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    ../autoeval_det/data/coco/val2014 \
    --answers-file \
    res/LLaVA-Lightning-MPT-7B-preview-detail-answer.jsonl
