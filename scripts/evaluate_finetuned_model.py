import glob
import re
import os
import cv2
import numpy as np
import argparse

from tqdm import tqdm

import third_parties.open_clip.src.open_clip as oc
import requests
import torch
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoModelForCausalLM
import collections
import csv
import pandas as pd
import ast
from collections import OrderedDict
from peft import PeftModel, PeftConfig

def read_csv(csv_file_path):
    return pd.read_csv(csv_file_path)

def generate_caption(model_name, model, processor, image_npy_path, bb, device, task_prompt=None, visualize=False, save=False, dest_dir=False):
    # rgb = np.load(image_npy_path)[:, :, :3]
    # image_npy_path = image_npy_path.replace("/projects/simca/extracted_dataset/postprocessed_dataset/gibson_randomGoal_coca_mask2former_test_postprocessed", "/media/tapicella/Win11_OS/Users/tapicella/Downloads/gibson_randomGoal_coca_mask2former_test_postprocessed_filtered_detection")
    rgb = np.load(image_npy_path, allow_pickle=True)['arr_0'].item()['image']
    im = Image.fromarray(rgb).convert('RGB')

    # Expand bounding box
    bbox_1 = [bb[0] - 10 if (bb[0] - 10) >= 0 else 0,
                bb[1] - 10 if (bb[1] - 10) >= 0 else bb[1],
                bb[2] + 10 if (bb[2] + 10) <= im.size[0] else bb[2],
                bb[3] + 10 if (bb[3] + 10) <= im.size[1] else bb[3]]
    im = im.crop(bbox_1)

    if model_name == "coca":
        inputs = processor(im).unsqueeze(0)
        inputs = inputs.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated_tokens = model.generate(inputs, generation_type='top_k')["text"]
        generated_text = oc.decode(generated_tokens[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
    elif model_name == "blip2":
        inputs = processor(im, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    elif model_name == "florence2":
        text_input = ""
        if task_prompt == "<REGION_TO_DESCRIPTION>":
            im = Image.fromarray(rgb).convert('RGB')
            text_input = "<loc_{}><loc_{}><loc_{}><loc_{}>".format(int(bbox_1[0]), int(bbox_1[1]), int(bbox_1[2]), int(bbox_1[3]))
            prompt = task_prompt + text_input
        else:
            prompt = task_prompt
        inputs = processor(text=prompt, images=im, return_tensors="pt").to('cuda', torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0].strip()
        generated_text = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(im.width, im.height)
        )
        generated_text = generated_text[task_prompt].split("<")[0]

    if visualize:
        img_vis = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        if task_prompt == "<REGION_TO_DESCRIPTION>":
            img_vis = cv2.rectangle(img_vis,(int(bbox_1[0]),int(bbox_1[1])),(int(bbox_1[2]),int(bbox_1[3])),(0,255,0),3)
        img_vis = cv2.resize(img_vis, (480, 480))
        cv2.imshow("RGB", img_vis)
        print(generated_text)
        cv2.waitKey(0)

    if save:
        img_vis = cv2.resize(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), (224, 224))
        cv2.imwrite(os.path.join(dest_dir, os.path.basename(image_npy_path).replace(".npz", ".png")), img_vis)
    return generated_text

def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path',
                        type=str,
                        default=None,
                        )
    parser.add_argument('--model_name',
                        type=str,
                        default="florence2" # blip2 coca
                        )
    parser.add_argument('--prompt',
                        type=str,
                        default="<REGION_TO_DESCRIPTION>" # None "<CAPTION>" "<REGION_TO_DESCRIPTION>"
                        )
    parser.add_argument('--test_file',
                        type=str,
                        default="/media/tapicella/Data/data/gibson_annotated_test_set.csv"
                        )
    parser.add_argument('--dest_file',
                        type=str,
                        default="/media/tapicella/Data/code/SImCa/res/a.csv"
                        )
    parser.add_argument('--visualize',
                        type=bool,
                        default=False
                        )
    parser.add_argument('--save_image',
                        type=bool,
                        default=False
                        )
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    ckpt_path = args.ckpt_path
    model_name = args.model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    task_prompt = args.prompt
    test_file = args.test_file
    dest_file = args.dest_file
    dest_dir = os.path.dirname(dest_file)
    visualize = args.visualize
    save = args.save_image

    if model_name == "coca":
        model, _, processor = oc.create_model_and_transforms(model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k")
    elif model_name == "blip2":
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    elif model_name == "florence2":
        model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', torch_dtype=torch.float16, trust_remote_code=True).to(
            device)
        processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)

    if ckpt_path is not None:
        if model_name == "coca":
            checkpoint = torch.load(ckpt_path)
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k.replace("module.","") # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        elif model_name == "blip2":
             model = PeftModel.from_pretrained(model, ckpt_path)
    model.to(device)
    model.eval()

    test_data = read_csv(test_file)

    with open(dest_file, 'w', newline='') as csvfile:
         writer = csv.writer(csvfile)
         writer.writerow(['filename', 'episode_id', 'object_id', 'bounding_box', 'proposed_caption', 'reference_caption'])
         for index, row in test_data.iterrows():
            filename = row.filename
            caption = generate_caption(model_name, model, processor, filename, ast.literal_eval(row.bounding_box), device, task_prompt, visualize, save, dest_dir)
            writer.writerow([filename, row.episode_id, row.object_id, row.bounding_box, caption, row.caption])
