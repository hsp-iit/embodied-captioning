""" This class is a wrapper for BLIP2 model in HuggingFace library.
    The wrapper is created to return both logits and text from the model. """

import torch
import cv2
import numpy as np

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
from torch.utils.data import DataLoader
from experimenting_env.captioner.captioning_predictor import CaptioningPredictor
from experimenting_env.captioner.captioning_datasets.base_dataset import CaptioningBaseDataset
from experimenting_env.captioner.utils.utils import Configuration


class BLIP2(CaptioningPredictor):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.processor = Blip2Processor.from_pretrained(cfg.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(cfg.model_name, load_in_8bit=True,
                                                                   device_map={"": 0},
                                                                   torch_dtype=torch.float16)

    def forward(self, inputs):
        inputs = self.processor(images=inputs, return_tensors="pt").to(self.device, torch.float16)
        outputs = self.model.generate(**inputs, output_logits=True, return_dict_in_generate=True)
        self.outputs["text"] = self.processor.batch_decode(outputs["sequences"], skip_special_tokens=True)[0].strip()
        self.outputs["logits"] = outputs["logits"]
        return self.outputs


if __name__ == '__main__':
    # Select device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    # Load model
    cfg = Configuration(arch_name='blip2', model_name='Salesforce/blip2-opt-2.7b', height=224, width=224)
    model = BLIP2(cfg.captioner)
    model.eval()
    model.to(device)

    # Load data
    images_dir = "..."
    test_dataset = CaptioningBaseDataset(
        images_dir,
        augmentation=None,
        preprocessing=None,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Inference
    for i, sample_batch in enumerate(tqdm(test_loader)):
        imgs = sample_batch['image']

        # Load to device
        imgs = imgs.to(device)

        # Inference
        output = model(imgs)

        # Load data
        image = sample_batch['rgb'].cpu().detach().numpy()[0].astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("RGB", image)
        print(output["text"])
        cv2.waitKey(0)
