python -m llava.serve.controller --host 10.140.37.1 --port 10000
python -m llava.serve.model_worker --host 10.140.37.1 --controller http://10.140.37.1:10000 --port 10002 --worker http://10.140.37.1:10002 --model-path liuhaotian/LLaVA-Lightning-MPT-7B-preview
python -m llava.serve.gradio_web_server --host 10.140.37.1 --controller http://10.140.37.1:10000
