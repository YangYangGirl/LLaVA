import json
import os
import subprocess
import tempfile
from nltk.translate.bleu_score import sentence_bleu
import torch
# import clip
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import evaluation

gpt_simplify = [json.loads(q) for q in open(os.path.expanduser('./res/gpt_simplify.jsonl'), "r")]
llava_concise = [json.loads(q) for q in open(os.path.expanduser('./res/LLaVA-Lightning-MPT-7B-preview-concise-answer.jsonl'), "r")]

output_file = open('./res/compare.jsonl', 'w')
cider = Cider()
for idx in range(0, len(llava_concise), 3):
    gts = gpt_simplify[idx//3]['simplify_text'].split(' ')
    gen = llava_concise[idx]['text'].split(' ')
    gts = [[i for i in gts if i != ',' and i != '.' and i != ', ']]
    gen = [i for i in gen if i != ',' and i != '.' and i != ', ']
    # score = bleu_metric.compute_score(gts, gen)
    
    bleu_1_score = sentence_bleu(gts, gen, weights=(1, 0, 0, 0))
    bleu_1_score = round(bleu_1_score, 5)
    llava_concise[idx]['bleu_1_score'] = bleu_1_score

    bleu_4_score = sentence_bleu(gts, gen, weights=(0, 0, 0, 1))
    bleu_4_score = round(bleu_4_score, 5)
    llava_concise[idx]['bleu_4_score'] = bleu_4_score

    print("simplified ", gts)
    print("llava concise version ", gen)
    print("bleu_1_score ", bleu_1_score)
    print("bleu_4_score ", bleu_4_score)
    output_file.write(json.dumps(llava_concise[idx]) + '\n')

output_file.close()

    