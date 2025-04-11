import os
import habitat
from experimenting_env.detector.dataset import (
    SinglecamEpisodeDetectionHabitatObjectsDataset,
    EpisodeFullDataset,
    PseudoFullDataset,
    SinglecamEpisodeFullDataset,
)
from experimenting_env.detector.model import *
from experimenting_env.utils.predictor_utils import Predictor, setup_cfg, MinimalPredictorWrapper
from experimenting_env.utils.train_helpers import get_loader, list_helper_collate, get_training_params, dict_helper_collate
from omegaconf import DictConfig
from experimenting_env.utils import triplet
from matplotlib import pyplot as plt
import torch
import numpy as np
from experimenting_env.sensor_data import BBSense
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.visualizer import ColorMode, Visualizer
from torch.optim import Adam
from torch import nn
from torch.functional import F
from detectron2.utils.events import EventStorage
from experimenting_env.detector.pseudolabeler import SoftConsensusLabeler, SemanticMapConsensusLabeler, SemanticMapConsensusCaptioner
import albumentations as A
import pickle
from experimenting_env.utils.detectron_utils import get_coco_item_dict
import hydra
import pytorch_lightning as pl
from experimenting_env.utils.predictor_utils import Captioner
import torch.distributed as dist
remap = {i:k for i, k in enumerate(BBSense.CLASSES)}

def load_dataset(path):
    dataset = SinglecamEpisodeFullDataset(path)
    print(f"Dataset loaded with {len(dataset)} samples")
    return dataset

def visualize_dataset(dataset, num_samples=51):
    for count in range(num_samples):
        x = dataset[count]
        metadata = MetadataCatalog.get('coco_2017_val')
        visualizer = Visualizer(
            x['image'].permute(1, 2, 0),
            metadata,
            instance_mode=ColorMode.IMAGE,
        )
        y = x['instances']
        y.pred_classes = torch.tensor([remap[p.item()] for p in y.gt_classes])
        y.pred_boxes = y.gt_boxes
        frame = visualizer.draw_instance_predictions(
            predictions=y.to('cpu')
        ).get_image()[:, :, ::-1]

        plt.imshow(frame)
        plt.show()
        
def load_pseudolabeler_loader(dataset):
    pseudolabel_loader = get_loader(
        dataset,
        shuffle=False,
        batch_size=8,
        collate_fn=dict_helper_collate,
    )
    
    return pseudolabel_loader

def get_pseudo_labeler(habitat_cfg):
    pseudolabeler = SemanticMapConsensusCaptioner(habitat_cfg)
    return pseudolabeler

def predict(trainer, pseudo_labeler, pseudolabel_loader):
    model_outs = trainer.predict(model=pseudo_labeler, dataloaders=pseudolabel_loader)
    return model_outs

@hydra.main(config_path='/mnt/storage/tgalliena/SImCa/confs/', config_name='train_policy.yaml')
def main(cfg):
    # clear hydra configuration
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    habitat_cfg = habitat.get_config(config_path=os.path.join(cfg.habitat_base_cfg_dir, cfg.habitat_cfg)) 
    dataset_path = habitat_cfg.pseudolabeler.input_folder
    dataset = load_dataset(dataset_path)
    # load the trainer for the model 
    trainer_config = get_training_params(cfg)
    trainer = pl.Trainer(accelerator='auto', devices='auto', **trainer_config)
    
    # load the pseudolabeler loader
    pseudolabel_loader = load_pseudolabeler_loader(dataset)
    
    # get the pseudolabeler
    pseudo_labeler = get_pseudo_labeler(habitat_cfg)
    # predict phase
    model_outs = predict(trainer, pseudo_labeler, pseudolabel_loader)
    
    #with open('/home/tommaso/Desktop/Workspace/SImCa/model_outs.pkl', 'rb') as f:
    #    model_outs = pickle.load(f)
        
    pseudo_labels = pseudo_labeler.get_pseudo_labels(model_outs, pseudolabel_loader)
    
    # Uncomment to visualize dataset
    #visualize_dataset(dataset)
    
    # Using multiple gpu in pl might leaves some process running in the background when the program exits 
    if dist.is_initialized():
        dist.destroy_process_group()    
        
    # TODO: Add further processing or analysis of pseudo_labels here

if __name__ == "__main__":
    main()
