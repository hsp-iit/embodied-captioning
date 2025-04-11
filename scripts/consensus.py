import csv
import cv2
import numpy as np
import open_clip
import os
import sys
import torch

from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore


clip_method = "torch" #"open_clip" # torch

# tf-idf
def compute_tf(word, document):
    return document.count(word) / len(document)

def compute_idf(word, documents):
    num_doc_contain_word = 0
    for d in documents:
        if word in d:
            num_doc_contain_word += 1
    
    return np.log( len(documents) / num_doc_contain_word)

def compute_tf_idf(word, document, documents):
    return compute_tf(word, document) * compute_idf(word, documents)

def rank_documents(caption, documents):
    scores = []
    for document in documents:
        tot = 0.0
        words = caption.split(" ")
        for word in words:
            tf_idf = compute_tf_idf(word, document, documents)
            tot += tf_idf
        score = tot / len(words)
        scores.append(score)

    return scores
# end tf-idf

if clip_method == "open_clip":
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
else:
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")

if len(sys.argv) < 2:
    img_name = "roi.png"
else:
    img_name = sys.argv[1]
img = cv2.imread(img_name)
img = cv2.resize(img, (224,224))[:,:,:3]

if len(sys.argv) < 3:
    captions_name = "captions.txt"
else:
    captions_name = sys.argv[2]
captions = [s.rstrip() for s in open(captions_name).readlines()]

clip_scores = []
with torch.no_grad():
    for caption in captions:
        if clip_method == "open_clip":
            image = preprocess(Image.fromarray(img)).unsqueeze(0)
            text = tokenizer(caption)
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            clip_score = torch.matmul(image_features, text_features.T).item()
        else:
            image = torch.tensor(img).unsqueeze(0)
            clip_score = metric(image, caption).numpy()
        clip_scores.append(clip_score)

# score captions using tf-idf, @TODO: check
tf_idf_scores = [np.mean(rank_documents(caption, captions)) for caption in captions]

print("------")
print("caption, clip score, tf-idf")
print("------")
for c, s, t in zip(captions, clip_scores, tf_idf_scores):
    print(c, "-", '{:.2f}'.format(s), '{:.4f}'.format(t))

print("------")
print("caption, combined score")
print("------")
mean_clip = np.mean(clip_scores)
std_clip = np.std(clip_scores)
for c, s, t in zip(captions, clip_scores, tf_idf_scores):
    s1 = (s - mean_clip) / std_clip
    print(c, "->", '{:.2f}'.format(s1 + t * 100))

filtered_captions = []
filtered_clip_scores = []
for c, s, t in zip(captions, clip_scores, tf_idf_scores):
    s1 = (s - mean_clip) / std_clip
    r = s1 + t * 100
    if r > 0.5:
        filtered_captions.append(c)
        filtered_clip_scores.append(s)
# score captions using tf-idf, @TODO: check
filtered_tf_idf_scores = [np.mean(rank_documents(caption, filtered_captions)) for caption in filtered_captions]
print("------")
print("filtered caption, final score")
print("------")
mean_clip = np.mean(filtered_clip_scores)
std_clip = np.std(filtered_clip_scores)
for c, s, t in zip(filtered_captions, filtered_clip_scores, filtered_tf_idf_scores):
    s1 = (s - mean_clip) / std_clip
    print(c, "->", '{:.2f}'.format(s1 + t * 100))
