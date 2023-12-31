srun -p INTERN2 --gres=gpu:8 --quotatype=reserved --ntasks=8 --ntasks-per-node=8 --cpus-per-task=5 python -u llava/train/train_mem.py \
    --model_name_or_path checkpoints/vicuna-7b-v1.3 \
    --version v1 \
    --data_path ./playground/data/llava_instruct_80k.json \
    --image_folder ../autoeval_det/data/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/LLaVA-Pretrained-Projectors/LLaVA-7b-pretrain-projector-v1-1-LCS-558K-blip_caption.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints/llava-7b-Lightning-finetune-0821 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
