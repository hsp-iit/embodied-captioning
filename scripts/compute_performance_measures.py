import nltk
import csv
import pandas as pd
import numpy as np
import ast
import os
import json
import argparse
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from torchmetrics.multimodal.clip_score import CLIPScore
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def calculate_weights(reference, proposed):
    ref_length = len(reference.split())
    proposed_length = len(proposed.split())
    max_n = min(ref_length, proposed_length, 4)  # Up to 4-grams

    # Example: Adjust weights based on the ratio of lengths
    weights = [1.0 / max_n] * max_n
    return weights


def compute_cosine_similarity(proposed_caption_list, reference_caption_list, model):
    # Encode the captions
    proposed_embeddings = model.encode(proposed_caption_list)
    reference_embeddings = model.encode(reference_caption_list)

    # Calculate cosine similarity matrix between proposed and reference, and get diagonal
    sentences_cosine_similarity = torch.diagonal(util.cos_sim(proposed_embeddings, reference_embeddings)).cpu().numpy()
    return sentences_cosine_similarity


def compute_score_pseudo_caption(test_dataset, annotated_dataset):
    results = []
    # metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with tqdm(total=len(test_dataset.keys())) as pbar:
        for key in test_dataset.keys():
            if key not in annotated_dataset:
                continue
            # CLIP score
            # npy_img = np.load(img_path, allow_pickle=True)['arr_0'].item()['image']
            # pil_img = Image.fromarray(npy_img)
            # cropped_img = pil_img.crop(ast.literal_eval(bbox))
            # tensor_img = pil_to_tensor(cropped_img)

            # clip_score = metric(tensor_img[:3, :, :], proposed)
            # BLEU Score
            reference = annotated_dataset[key]
            proposed = test_dataset[key]['pseudocaption']
            if proposed is None:
                print("None")
                continue
            weights = calculate_weights(reference, proposed)
            bleu = sentence_bleu([reference], proposed, weights=weights)

            # METEOR Score
            meteor = meteor_score([reference.split()], proposed.split())

            # sentence bert
            embeddings = model.encode([proposed, reference])
            sentence_similarity = model.similarity(embeddings, embeddings)[0][1].item()

            # ROUGE Score
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(reference, proposed)
            rouge1 = rouge_scores['rouge1'].fmeasure
            rouge2 = rouge_scores['rouge2'].fmeasure
            rougeL = rouge_scores['rougeL'].fmeasure

            formatted_bleu = round(bleu, 4)
            formatted_meteor = round(meteor, 4)
            formatted_rouge1 = round(rouge1, 4)
            formatted_rouge2 = round(rouge2, 4)
            formatted_rougeL = round(rougeL, 4)
            # formatted_clip = round(clip_score.item(), 4)

            formatted_sentence_similarity = round(sentence_similarity, 4)

            # Store the results
            results.append({
                'Reference Caption': reference,
                'Proposed Caption': proposed,
                'BLEU Score': formatted_bleu,
                'METEOR Score': formatted_meteor,
                'ROUGE-1': formatted_rouge1,
                'ROUGE-2': formatted_rouge2,
                'ROUGE-L': formatted_rougeL,
                'Sentence Similarity': formatted_sentence_similarity
            })
            pbar.update(1)

    # Create a DataFrame from results
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv(dst_path, index=False)

    keys = ["Stat"] + list(results_df.columns[2::])
    stats_df = pd.DataFrame(columns=keys)
    stats_df["Stat"] = ["Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"]
    for key in keys[1::]:
        column = np.array(results_df[key])
        min_val = np.amin(column)
        percentiles = np.percentile(column, [25, 50, 75])
        q1 = percentiles[0]
        q2 = percentiles[1]
        q3 = percentiles[2]
        median = np.median(column)
        mean_val = round(column.mean(), 4)
        std_dev = round(column.std(), 4)
        max_val = np.amax(column)
        assert q2 == median, "Median wrong!"

        stats_df[key] = [mean_val, std_dev, min_val, q1, q2, q3, max_val]

        print("Mean {}: {}".format(key, mean_val))
        print("Standard deviation {}: {}".format(key, std_dev))
        print("Min {}: {}".format(key, min_val))
        print("Q1 {}: {}".format(key, q1))
        print("Q2 {}: {}".format(key, q2))
        print("Q3 {}: {}".format(key, q3))
        print("Max {}: {}".format(key, max_val))
        print("=============")
    stats_df.to_csv(dst_path.replace(".csv", '_stats.csv'), index=False)
    print("Results saved!")

def compute_score(image_path_list, bbox_list, reference_caption_list, proposed_caption_list, dst_path):
    results = []
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    SBERT_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device="cuda")

    # Cosine similairty score of SBERT embeddings
    cosine_score = compute_cosine_similarity(proposed_caption_list, reference_caption_list, SBERT_model)

    for i, (img_path, bbox, reference, proposed) in enumerate(zip(image_path_list, bbox_list, reference_caption_list,
                                                                  proposed_caption_list)):
        # CLIP score
        # npy_img = np.load(img_path)
        npy_img = np.load(img_path, allow_pickle=True)['arr_0'].item()['image']
        pil_img = Image.fromarray(npy_img)
        cropped_img = pil_img.crop(ast.literal_eval(bbox))
        tensor_img = pil_to_tensor(cropped_img)
        clip_score = metric(tensor_img[:3, :, :], proposed)

        # BLEU Score
        weights = calculate_weights(reference, proposed)
        bleu = sentence_bleu([reference], proposed, weights=weights)

        # METEOR Score
        meteor = meteor_score([reference.split()], proposed.split())

        # ROUGE Score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, proposed)
        rouge1 = rouge_scores['rouge1'].fmeasure
        rouge2 = rouge_scores['rouge2'].fmeasure
        rougeL = rouge_scores['rougeL'].fmeasure

        formatted_bleu = round(bleu, 4)
        formatted_meteor = round(meteor, 4)
        formatted_rouge1 = round(rouge1, 4)
        formatted_rouge2 = round(rouge2, 4)
        formatted_rougeL = round(rougeL, 4)
        formatted_clip = round(clip_score.item(), 4)
        formatted_cosine = round(cosine_score[i].item(), 4)

        # Store the results
        results.append({
            'Reference Caption': reference,
            'Proposed Caption': proposed,
            'BLEU Score': formatted_bleu,
            'METEOR Score': formatted_meteor,
            'ROUGE-1': formatted_rouge1,
            'ROUGE-2': formatted_rouge2,
            'ROUGE-L': formatted_rougeL,
            'CLIP Score': formatted_clip,
            'Cosine similarity Score': formatted_cosine
        })

    # Create a DataFrame from results
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv(dst_path, index=False)

    keys = ["Stat"] + list(results_df.columns[2::])
    stats_df = pd.DataFrame(columns=keys)
    stats_df["Stat"] = ["Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"]
    for key in keys[1::]:
        column = np.array(results_df[key])
        min_val = np.amin(column)
        percentiles = np.percentile(column, [25, 50, 75])
        q1 = percentiles[0]
        q2 = percentiles[1]
        q3 = percentiles[2]
        median = np.median(column)
        mean_val = round(column.mean(), 4)
        std_dev = round(column.std(), 4)
        max_val = np.amax(column)
        assert q2 == median, "Median wrong!"

        stats_df[key] = [mean_val, std_dev, min_val, q1, q2, q3, max_val]

        print("Mean {}: {}".format(key, mean_val))
        print("Standard deviation {}: {}".format(key, std_dev))
        print("Min {}: {}".format(key, min_val))
        print("Q1 {}: {}".format(key, q1))
        print("Q2 {}: {}".format(key, q2))
        print("Q3 {}: {}".format(key, q3))
        print("Max {}: {}".format(key, max_val))
        print("=============")
    stats_df.to_csv(dst_path.replace(".csv", '_stats.csv'), index=False)
    print("Results saved!")

def read_csv(csv_path):
    dataset = pd.read_csv(csv_path)
    dataset_cleand = dataset.dropna(subset=['reference_caption'])

    return dataset_cleand.filename.tolist(), dataset_cleand.bounding_box.tolist(), dataset_cleand.reference_caption.tolist(), dataset_cleand.proposed_caption.tolist()


def read_json(json_path):
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    return dataset


def read_csv_caption(csv_path):
    dataset = pd.read_csv(csv_path)
    new_d = {}
    for i in range(len(dataset)):
        row = dataset.iloc[i]
        episode_id = row.episode_id
        object_id = row.object_id
        new_d[f'({episode_id}, {object_id})'] = row.caption
    return new_d


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path',
                        type=str,
                        default="...",
                        )
    parser.add_argument('--json_path',
                        type=str,
                        default=None,
                        )
    parser.add_argument('--dst_path',
                        type=str,
                        default="...",
                        )
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    # nltk.download('wordnet')
    csv_path = args.csv_path
    json_path = args.json_path
    dst_path = args.dst_path

    if json_path is None:
        # read annotated dataset
        image_path_list, bbox_list, reference_caption_list, proposed_caption_list = read_csv(csv_path)
        compute_score(image_path_list, bbox_list, reference_caption_list, proposed_caption_list, dst_path)
    else:
        test_dataset = read_json(json_path)
        annotated_dataset = read_csv_caption(csv_path)
        # read annotated dataset
        compute_score_pseudo_caption(test_dataset, annotated_dataset)


