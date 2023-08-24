GPUS=1 srun -p INTERN2 --gres=gpu:1 -w SH-IDC1-10-140-37-4 python -m llava.serve.controller --host 0.0.0.0 --port 10071
GPUS=1 srun -p INTERN2 --gres=gpu:1 -w SH-IDC1-10-140-37-4 python -m llava.serve.model_worker --host 0.0.0.0 --controller http://10.140.37.4:10071 --port 10072 --worker http://localhost:10072 --model-path liuhaotian/LLaVA-Lightning-MPT-7B-preview
GPUS=1 srun -p INTERN2 --gres=gpu:1 -w SH-IDC1-10-140-37-4 python -m llava.serve.gradio_web_server --share --port 10073 --host 0.0.0.0 --controller http://10.140.37.4:10071
