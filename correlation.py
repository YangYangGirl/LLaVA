import os
import json
import shutil
data = [json.loads(q) for q in open(os.path.expanduser("playground/data/coco2014_val_qa_eval/qa90_questions.jsonl"), "r")]
for idx in range(0, len(data), 3):
    shutil.copy('../autoeval_det/data/coco/val2014/COCO_val2014_' + data[idx]['image'], 'playground/data/sampled_imgs/' + str(idx) + '_' + data[idx]['image'])
# import pdb; pdb.set_trace()