import numpy as np
import torch
import glob
import json
import re
import os
import argparse
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms import functional 
from tqdm import tqdm
import pickle
import cv2
#from lavis.models import load_model_and_preprocess

class PseudoCaptioner:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_id = model_id
        self.access_token = "*"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_weights_dir = "/work/tgalliena/Workspace/eco_score/model_weights"
        self.args = self.parse_arguments()
        self.get_bbs_path_list()
        self.group_captions()
        #with open('/work/tgalliena/SImCa/train_grouped_filtered_box.pickle', 'rb') as f:
        #    self.grouped_captions = pickle.load(f)
        
    def get_bbs_path_list(self):
        self.bbs_path_list = glob.glob(self.args.file_path + '/*.npz')
        
    def _setup_blip2_model(self):
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(
            "blip2_image_text_matching", "coco", device=self.device, is_eval=True
        )
        
    def _setup_clip_model(self):
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                                            cache_dir="/work/tgalliena/huggingface_cache",
                                                            device_map="auto")
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                                    cache_dir="/work/tgalliena/huggingface_cache",
                                                    device_map="auto")

    def _setup_llm_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.access_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            token=self.access_token,
            device_map="auto",
            cache_dir="/work/tgalliena/huggingface_cache"
        )
        self.model.eval()
        
    def save_pseudocaptions(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.pseudocaptions, f)

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--file_path", 
            type=str,
            help="Path to the input bbs or bbsgt files",
            required=True
        )
        parser.add_argument(
            "--output_csv_path",
            type=str, 
            help="Path to the output CSV file",
            required=True
        )
        parser.add_argument(
            "--method",
            type=str,
            choices=["llm", "clip", "blip2_itm", 'blip2_itc', 'mobileclip', 'openclip'],
            help="Method to use to compute the pseudo-captions(either 'llm' or 'clip')",
            required=True
        )
        args = parser.parse_args()
        return args
    
    def filter_captions(self, caption):
        
        banned_words = [
            # Living Beings
            "person", "man", "woman", "boy", "girl", "child", "children", "adult", "kid", "baby", "human", "people", "group", "crowd",
            "dog", "cat", "bird", "fish", "horse", "animal", "pet", "elephant", "lion", "tiger", "monkey", "mouse", "rabbit", "cow",
            "pig", "sheep", "deer", "bear", "chicken", "duck", "goat", "camel", "snake", "frog", "turtle", "whale", "dolphin", "insect",
            "bug", "spider",
            
            # Image Quality or Context
            "blurry", "picture", "image", "photo", "portrait", "painting", "drawing", "sketch", "screenshot", "artwork", "filter", "3d",
            "rendering", 
            # Generic/Non-descriptive Terms
            "thing", "stuff", "object", "item", "something", "stuff", "device", "equipment", "material", "machine", "gadget",
            "unknown", "unidentified", "indistinguishable", "living room", "kitchen", "bedroom", "bathroom", "dining room", "living room", "room",
            
            # Non-Indoor Terms (Optional)
            "car", "vehicle", "bike", "truck", "street", "road", "tree", "forest", "mountain", "park", "outdoor", "sky", "landscape",
            "scenery",
            
            # Action Words (Optional)
            "running", "jumping", "walking", "talking", "playing", "sitting", "standing", "moving", "holding", "eating", "drinking",
            "flying", "swimming", "driving"
        ]
        
        if not any(banned_word.lower() in caption.lower() for banned_word in banned_words):
            return True
        return False

    def group_captions(self):
        """Create a dictionary of captions grouped by (episode_id, object_id) tuples.

        Args:
            bbs_path_list (list[String]): List of bbs or bbsgt file paths
        """
        grouped_captions = {}
        for bbs_path in tqdm(self.bbs_path_list):
            bbs = np.load(bbs_path, allow_pickle=True)['arr_0'].item()
            instances = bbs['instances']       
            rgb = bbs['image']
            for idx in range(len(instances)):
                key = (instances.infos[idx]['id_episode'], instances.infos[idx]['id_object'])
                data = {
                        'filename': bbs_path,
                        'image': rgb,
                         'pred_box': instances.pred_boxes[idx].tensor[0].numpy(),
                         'info': instances.infos[idx],
                         'caption': instances.captions[idx]
                        }
                if self.filter_captions(data['caption']):
                    if key not in grouped_captions:
                        grouped_captions[key] = [data]
                    else:
                        grouped_captions[key].append(data)
                    
        self.grouped_captions = grouped_captions
        
        with open('/work/tgalliena/SImCa/val_grouped_filtered_box.pickle', 'wb') as f:
            pickle.dump(grouped_captions, f)

    def compute_captions_frequency(self):
        """Compute frequency of captions for each object.

        Args:
            grouped_captions: Dictionary mapping (episode_id, object_id) tuples to lists of instance data.

        Returns:
            Dictionary mapping (episode_id, object_id) tuples to lists of [frequency, caption] pairs.
        """
        id2captions_freq = {}
        
        for key in self.grouped_captions:
            captions_freq_dict = {}
            
            for instance in self.grouped_captions[key]:
                caption = instance['caption']
                captions_freq_dict[caption] = captions_freq_dict.get(caption, 0) + 1
                    
            captions_freq_list = [[freq, caption] for caption, freq in captions_freq_dict.items()]
            id2captions_freq[key] = captions_freq_list
            
        return id2captions_freq
    
    def expand_box(self, box, expand_factor, image_size):
        x1, y1, x2, y2 = box  # Assuming box is a Boxes object from detectron2
        width, height = image_size[0], image_size[1]
        
        box_width = x2 - x1
        box_height = y2 - y1
        
        new_x1 = int(max(x1 - expand_factor * box_width, 0))
        new_y1 = int(max(y1 - expand_factor * box_height, 0))
        new_x2 = int(min(x2 + expand_factor * box_width, width))
        new_y2 = int(min(y2 + expand_factor * box_height, height))
            
        return new_x1, new_y1, new_x2, new_y2
    
    @torch.no_grad()
    def itc_score(self, image_embeddings, text_ids, text_atts):
        """
        Calculate ITC scores for given images and captions.

        Parameters:
        - model: blip2 model
        - image_embedding: Image embeddings
        - text_ids: Tokenized text
        - text_atts: Attention mask for text_ids
        """

        device = image_embeddings[0].device
        #image_embedding = image_embedding.repeat(text_ids.shape[0], 1, 1).to(device)
        image_atts = torch.ones(image_embeddings.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.model.query_tokens.expand(image_embeddings.shape[0], -1, -1)
        query_output = self.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeddings,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_feats = F.normalize(self.model.vision_proj(query_output.last_hidden_state), dim=-1)

        text_output = self.model.Qformer.bert(
            text_ids,
            attention_mask=text_atts,
            return_dict=True,
        )
        text_feat = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        sims = torch.bmm(image_feats, text_feat.unsqueeze(-1))
        sim, _ = torch.max(sims, dim=1)

        return sim.cpu().squeeze().numpy().item()

    
    @torch.no_grad()
    def itm_score(self, image_embedding, text_ids, text_atts):
        """
        Calculate ITM scores for given images and captions.

        Parameters:
        - model: blip2 model
        - image_embedding: Image embeddings
        - text_ids: Tokenized text
        - text_atts: Attention mask for text_ids
        """
        device = image_embedding.device
        #image_embedding = image_embedding.repeat(text_ids.shape[0], 1, 1).to(device)
        image_atts = torch.ones(image_embedding.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.model.query_tokens.expand(image_embedding.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
        attention_mask = torch.cat([query_atts, text_atts], dim=1)

        output_itm = self.model.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embedding,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.model.itm_head(vl_embeddings).mean(dim=1)
        itm_logit = torch.nn.functional.softmax(itm_logit, dim=-1)[:, 1]
        
        return itm_logit.cpu().numpy().item()
    
    @torch.no_grad()
    def blip2_score(self):
        """
        Calculate blip2 (ITC, ITM) scores for given images and captions.
        Parameters:
        - model: blip2 model
        - vis_processors: Image processors for blip2 model
        - captions (DataFrame): DataFrame where columns are filenames and values are captions.
        - device: Device to run the model on.
        """
        
        result_dict = {}
        
        for key in tqdm(self.grouped_captions):
            captions_list = []
            images_list = []
            scores_list = []
            
            for instance in self.grouped_captions[key]:
                image = self.preprocess_image(instance)

                image = (
                    self.vis_processors["eval"](functional.to_pil_image(image))
                    .unsqueeze(0)
                    .to(self.model.device)
                )
                image_embedding = self.model.ln_vision(self.model.visual_encoder(image)).float()
                
                text = self.model.tokenizer(instance['caption'], truncation=True, padding=True, max_length=32, return_tensors="pt").to(
                    self.model.device
                )
                if self.args.method[-3:] == "itc":
                    score = self.itc_score(image_embedding, text.input_ids, text.attention_mask)
                elif self.args.method[-3:] == "itm":
                    score = self.itm_score(image_embedding, text.input_ids, text.attention_mask)
                
                captions_list.append(instance['caption'])
                scores_list.append(score)
                    
            result_dict[str(key)] = {"captions": captions_list, "scores": scores_list}
                
        score_file_path = self.args.output_csv_path
        print("Scoring completed. Saving scores to", score_file_path)
        with open(score_file_path, "w") as f:
            json.dump(result_dict, f)
            
    @torch.no_grad()
    def eco_clip_score(self):
        """
        Calculate CLIP scores for given images and captions.
        """
        result_dict = {}

        for key in tqdm(self.grouped_captions):
            caption_list = []
            clip_scores_list = []

            for instance in self.grouped_captions[key]:

                image = self.preprocess_image(instance)
                image = self.preprocess(functional.to_pil_image(image)).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                caption_list.append(instance['caption'])
                
                tokenized_caption = self.tokenizer(instance['caption']).to(self.device)
                caption_features = self.model.encode_text(tokenized_caption)
                caption_features /= caption_features.norm(dim=-1, keepdim=True)

                clip_score = (image_features @ caption_features.T).detach().cpu().tolist()[0][0]
                clip_scores_list.append(clip_score)
                
            result_dict[str(key)] = {"captions": caption_list, "scores": clip_scores_list}

        score_file_path = self.args.output_csv_path
        print("Scoring completed. Saving scores to", score_file_path)
        with open(score_file_path, "w") as f:
            json.dump(result_dict, f)
            

            
    def preprocess_image(self, instance):
        x1, y1, x2, y2 = self.expand_box(instance['pred_box'], 0.1, (1280, 1280))
        image = instance['image'][y1:y2, x1:x2, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
        
    def compute_clip_score(self, image, caption):
        inputs = self.clip_processor(images=image, text=caption, return_tensors="pt", padding=True).to('cuda:0')
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        
        return logits_per_image

    def extract_caption_from_llm_output(self, text):
        pattern = r'<Caption>(.*?)</Caption>'
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def generate_llm_caption(self, captions_freq_list):
        prompt = f"""
You are an advanced language model tasked with generating a concise and accurate caption for an object. You are given a list of captions along with their frequencies. Each caption may represent a different viewpoint and might not always be accurate. Your goal is to generate a single, coherent caption that accurately describes the main object, based on the provided information. The generated caption should not exceed 20 words and must be encapsulated within <Caption> ... </Caption> tags.
Consider that all the caption are of indoor objects and are from static sceene without any kind of living beings (no humans, no people and animals, no man etc. etc.) so you MUST NOT use this kind of worlds in the captions . Also consider that the image can have low quality but you HAVE not to include sentences like "A blurry image of ...", "A picture of ...", "A portrait of ...", " A painting of ..." ecc. ecc.
Here is the format of the input you will receive:
[[frequency, "caption"]]

And some examples of the task you will be performing:

Example Input:
[[5, "A red apple on a table"], [3, "A shiny red apple"], [1, "A red fruit"], [2, "A red apple"]]

Example Output:
<Caption>A shiny red apple on a table</Caption>

Example Input:
[[8, "A blurry image of wooden table"], [3, "A cat on a wooden table"], [15, "A brown dog"]]

Example Output:
<Caption>A wooden table</Caption>

Example Input:
[[6, "A blue car parked on the street"], [4, "A car"], [2, "A blue vehicle"], [1, "A car on the street"]]

Example Output:
<Caption>A blue car parked on the street</Caption>

Example Input:
[[7, "A person standing in a room"], [85, "A picture of a brown couch with some pillows"], [17, "A blurry image of a cat on a brown couch"]]

Example Output:
<Caption>A brown couch with some pillows</Caption>

Example Input:
[[5, "A wooden table with a plate on it"], [2, "A table with a plate and a couch in the room"], [3, "A wooden table"], [1, "A plate on a wooden table"]]

Example Output:
<Caption>A wooden table with a plate on it</Caption>

Your Task:
1. Analyze the provided list of captions and their frequencies.
2. Synthesize an accurate caption that reflects the most reliable and frequent details.
3. Ensure the generated caption describes only the main objects and mentions other objects only in relation to the main object.
4. Ensure the generated caption is no longer than 20 words.
5. Encapsulate your generated caption within <Caption> ... </Caption> tags.

Input:
{str(captions_freq_list)}

Output:
"""

        messages = [
            {
                "role": "system",
                "content": prompt
            },
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

        response = outputs[0][input_ids.shape[-1]:]
        generated_pseudo_caption = self.tokenizer.decode(response, skip_special_tokens=True)
        
        return self.extract_caption_from_llm_output(generated_pseudo_caption)

    def compute_llm_pseudo_captions(self):
        self._setup_llm_model()
        
        id2captions_freq = self.compute_captions_frequency()
        
        self.pseudocaptions = dict()

        for obj_id in tqdm(id2captions_freq):
            torch.cuda.empty_cache()
            pseudocaption = self.generate_llm_caption(id2captions_freq[obj_id])
            self.pseudocaptions[str(obj_id)] = {'captions_list': id2captions_freq[obj_id], 'pseudocaption': pseudocaption}
        
        self.save_pseudocaptions(self.args.output_csv_path)
        
    def compute_clip_pseudo_captions(self):
        self._setup_clip_model()
        
        id2clip_score = dict()
        
        for obj_id in self.grouped_captions:
            id2clip_score[obj_id] = []
            for instance in self.grouped_captions[obj_id]:
                image = self.preprocess_image(instance)
                caption = instance['caption']
                score = self.compute_clip_score(image, caption)
                id2clip_score[obj_id].append([score.item(), caption])
                
        self.pseudocaptions = dict()
                
        for obj_id in id2clip_score:
            id2clip_score[obj_id].sort(key=lambda x: x[0], reverse=True)
            score, pseudocaption = id2clip_score[obj_id][0] 
            self.pseudocaptions[str(obj_id)] = {'captions_list': id2clip_score[obj_id], 'pseudocaption': [score, pseudocaption]}
        
        self.save_pseudocaptions(self.args.output_csv_path)
        
    def compute_pseudo_captions(self):
        if self.args.method == "llm":
            self.compute_llm_pseudo_captions()
        elif self.args.method == "clip":
            self.compute_clip_pseudo_captions()
        elif self.args.method[:5] == "blip2":
            self._setup_blip2_model()
            self.blip2_score()
        elif self.args.method == "mobileclip":
            from mobileclip import create_model_and_transforms, get_tokenizer
            self.model, _, self.preprocess = create_model_and_transforms(
                "mobileclip_b",
                pretrained=os.path.join(self.model_weights_dir, "mobileclip", "mobileclip_blt.pt"),
                device=self.device,
            )
            self.tokenizer = get_tokenizer("mobileclip_b")
            self.eco_clip_score()
        elif self.args.method == "openclip":
            import open_clip

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-bigG-14")
            self.eco_clip_score()
            
if __name__ == '__main__':
    pseudocaptioner = PseudoCaptioner()
    pseudocaptioner.compute_pseudo_captions()
    
