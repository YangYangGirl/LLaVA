import json
import os
from pycocoevalcap.bleu.bleu import Bleu

gpt_simplify = [json.loads(q) for q in open(os.path.expanduser('./res/gpt4_simplify.jsonl'), "r")]
llava_concise = [json.loads(q) for q in open(os.path.expanduser('./res/LLaVA-Lightning-MPT-7B-preview-concise-answer.jsonl'), "r")]

for idx, line in enumerate(llava_concise, 3):

    score = Bleu(gpt_simplify[idx//3], llava_concise[idx])

    import pdb; pdb.set_trace()