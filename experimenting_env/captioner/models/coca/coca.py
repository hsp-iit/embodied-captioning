""" This class is a wrapper for CoCa model in OpenCLIP library.
    The wrapper is created to return both logits and text from the model. """
import torch
import cv2
import open_clip
import albumentations as albu
import numpy as np
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from experimenting_env.captioner.captioning_predictor import CaptioningPredictor
from experimenting_env.captioner.captioning_datasets.base_dataset import CaptioningBaseDataset
from experimenting_env.captioner.utils.utils import Configuration
from experimenting_env.captioner.models.coca.factory import create_model_and_transforms


class CoCa(CaptioningPredictor):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.model, _, self.preprocess = create_model_and_transforms(cfg.model_name,
                                                                     pretrained=cfg.checkpoint_name, 
                                                                     cache_dir="/mnt/storage/tgalliena/model_cache")
        self.tokenizer = open_clip.get_tokenizer(cfg.model_name)

    def forward(self, inputs):
        inputs = self.preprocess(inputs).unsqueeze(0).to(self.device)
        outputs = self.model.generate(inputs, generation_type='top_k')
        self.outputs["text"] = open_clip.decode(outputs["text"][0]).split("<end_of_text>")[0].replace("<start_of_text>",
                                                                                                      "")
        self.outputs["logits"] = outputs["logits"]
        return self.outputs


if __name__ == '__main__':
    # Select device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    # Load model
    cfg = Configuration(arch_name='coca', model_name='coca_ViT-L-14',
                        checkpoint_name='mscoco_finetuned_laion2B-s13B-b90k', height=224, width=224)
    model = CoCa(cfg=cfg.captioner)
    model.eval()
    model.to(device)

    # Load data
    images_dir = "/home/tapicella/Downloads/temp_data/rgb"

    test_dataset = CaptioningBaseDataset(
        images_dir,
        augmentation=None,
        preprocessing=None,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    augmentation = albu.Compose([albu.Resize(height=224, width=224, p=1)])
    transform = T.ToPILImage()

    # Inference
    for i, sample_batch in enumerate(tqdm(test_loader)):
        imgs = sample_batch['image']

        # Load to device
        imgs = imgs.to(device)

        # Workaround to manage low memory and the fact that albumentation works on numpy
        inputs_tensor = imgs.clone()
        inputs = imgs.clone()
        inputs = inputs.item() if inputs.device == "cpu" else inputs.cpu().detach().numpy()
        # inputs_final = torch.zeros(inputs_tensor.shape[0], inputs_tensor.shape[3], 224, 224)
        inputs_final = None

        # Preprocess each element in the batch
        for ind, x in enumerate(inputs):
            x = augmentation(image=x)["image"]
            # inputs_final[ind] = torch.from_numpy(x).permute(2,1,0)
            inputs_final = Image.fromarray(x)

        # Inference
        output = model(inputs_final)

        # Load data
        image = sample_batch['rgb'].cpu().detach().numpy()[0].astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("RGB", image)
        print(output["text"])
        cv2.waitKey(0)
