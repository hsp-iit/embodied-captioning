from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import open_clip
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import ImageList, Instances, Boxes
from torch import nn
from torchmetrics.detection import MeanAveragePrecision as MAP

from experimenting_env.captioner.utils.utils import Configuration
from experimenting_env.captioner.models.coca.coca import CoCa
from experimenting_env.captioner.models.blip2.blip2 import BLIP2
from experimenting_env.sensor_data import BBSense
from experimenting_env.utils.detectron_utils import (
    enable_dropout,
    fast_rcnn_inference,
)
from experimenting_env.utils.roi_head_wrappers import MinimalPredictorWrapper
import logging
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultPredictor
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import cv2


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_semantic_map(classes, mask):

    size = mask.shape[1:]
    segm = np.zeros(size)

    for i in range(len(mask)):
        segm[mask[i].bool()] = classes[i]

    segm[mask.sum(0) == 0] = len(BBSense.CLASSES)  # background

    return segm


def get_gt_mask(bbs: Instances, compact_mask):
    size = bbs.image_size
    segm = np.zeros((len(bbs), *size))

    for i in range(len(bbs)):
        segm[i] = bbs.gt_classes[i]

    segm[bbs.gt_masks.sum(0) == 0] = len(BBSense.CLASSES)  # background

    return segm


def load_predictor(args):
    cfg = setup_cfg(args)
    return Predictor(cfg)


def setup_cfg(args):
    # load config from file and command-line arguments

    if isinstance(args, CfgNode):
        cfg = args
    else:
        cfg = get_cfg()

    cfg.MODEL.DEVICE = "cpu"
    if hasattr(args, "config_file"):
        cfg.merge_from_file(args.config_file)

    if hasattr(args, "opts"):
        cfg.merge_from_list(args.opts)

    if hasattr(args, "confidence_threshold"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold

    return cfg

def IoU(box1, box2):
    
    x1, y1, a1, b1 = box1
    x2, y2, a2, b2 = box2
    
    area1 = (a1 - x1)*(b1 - y1)
    area2 = (a2 - x2)*(b2 - y2)
    
    xx = max(x1, x2)
    yy = max(y1, y2)
    aa = min(a1, a2)
    bb = min(b1, b2)
    
    w = max(0, aa - xx)
    h = max(0, bb - yy)
    
    intersection_area = w*h
    
    union_area = area1 + area2 - intersection_area
    
    IoU = intersection_area / union_area
    
    return IoU

def compute_all_ious(bboxes):
    """
    Compute the IoU for every pair of bounding boxes.

    Args:
        bboxes: List of bounding boxes, each defined as [x1, y1, x2, y2].

    Returns:
        A 2D NumPy array containing IoU values.
    """
    n = len(bboxes)
    iou_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):  # Optimize by avoiding redundant computations
            iou = IoU(bboxes[i], bboxes[j])
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou  # Symmetric

    return iou_matrix

def get_high_iou_pairs(iou_matrix, threshold):
    """
    Get indices of bounding box pairs with IoU above the threshold.

    Args:
        iou_matrix: A 2D NumPy array containing IoU values.
        threshold: The IoU threshold.

    Returns:
        A list of tuples, where each tuple contains the indices (i, j) of bounding boxes
        with IoU > threshold.
    """
    pairs = []
    n = iou_matrix.shape[0]

    for i in range(n):
        for j in range(i + 1, n):  # Avoid self-comparison and duplicate pairs
            if iou_matrix[i, j] > threshold:
                pairs.append((i, j))

    return pairs

def select_element_to_discard(scores, pairs):
    discarded_element_idx = []
    
    for idx1, idx2 in pairs:
        if scores[idx1] < scores[idx2]:
            discarded_element_idx.append(idx1)
        else:
            discarded_element_idx.append(idx2)

    return discarded_element_idx

class Captioner(pl.LightningModule):
    def __init__(self, cfg=None, input_format=None, load_checkpoint=False, checkpoint_path=None, metadata=None, model=None):
        super().__init__()
        
        if model is None:
            self.model = self.get_captioner(cfg.captioner) 
        else:
            self.model = model
            
        # Set the model to evaluation model
        self.model.eval()
        
        assert self.model is not None, "No model provided"
        
        logger.info("Captioner model loaded successfully")
        
        if load_checkpoint and checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            logger.info(f"Captioner model checkpoint loaded successfully from {checkpoint_path}")
            
        self.model.to('cpu')
        #self.model.to('cuda:0')

    def get_captioner(self, cfg):
        """ Get the captioner model based on the configuration settings"""
        if cfg.arch_name == "coca": 
            captioner_cfg = Configuration(arch_name=cfg.arch_name, model_name=cfg.model_name,
                                            checkpoint_name=cfg.checkpoint_name, height=cfg.height, width=cfg.width) 
            captioner = CoCa(captioner_cfg.captioner)

        elif cfg.arch_name == "blip2":
            captioner_cfg = Configuration(arch_name=cfg.arch_name, model_name=cfg.model_name,
                                            height=cfg.height, width=cfg.width) 
            captioner = BLIP2(captioner_cfg.captioner)

        return captioner.eval()
            
    def forward(self, x):       
        out = self.model(x)            
        caption = out['text'] 
        
        return caption

def build_mask2former():
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-instance")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance")
    
    return model, processor

class Predictor(pl.LightningModule):
    

    def __init__(self, cfg=None, input_format=None, load_checkpoint=True, metadata=None, model=None):
        super().__init__()

        # if model is None and cfg is not None:
        #     if isinstance(cfg, CfgNode) is False:
        #         cfg = setup_cfg(cfg)
        #     self.cfg = cfg.clone()  # cfg can be modified by model

        #     model = build_model(self.cfg)
        #     if load_checkpoint:
        #         checkpointer = DetectionCheckpointer(model)
        #         checkpointer.load(cfg.MODEL.WEIGHTS)
        # elif model is None and cfg is None:
        #     model = model_zoo.get(
        #         "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        #         trained=load_checkpoint,
        #         device='cpu',
        #     )
        model, processor = build_mask2former()
        assert model is not None, "No model provided"
        self.model = model
        self.processor = processor

        self.test_map_metric = MAP(class_metrics=True)
        self.model.eval()
        #self.model.to('cpu')
        
    def on_test_epoch_end(self):

        results = self.test_map_metric.compute()
        for k in results.keys():
            if k == "map_per_class":
                for i, v in enumerate(BBSense.CLASSES.values()):
                    self.log(
                        f"test_{k}_{v}",
                        results[k][i],
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )
            else:
                self.log(
                    f"test_{k}",
                    results[k],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):

        self.model.eval()

        predictions = self(batch)[0]

        gt = [
            {
                'boxes': b['instances'].gt_boxes.tensor,
                'labels': b['instances'].gt_classes.int(),
                "masks": b['instances'].gt_masks.tensor,
            }
            for b in batch
        ]
        pred = [
            {
                'boxes': b['instances'].pred_boxes.tensor,
                'labels': b['instances'].pred_classes,
                'scores': b['instances'].scores,
                'masks': b['instances'].pred_masks,
            }
            for b in predictions
        ]

        self.test_map_metric.update(pred, gt)

    def set_head_wrapper(self, head_class: MinimalPredictorWrapper):
        """We implement custom ROIHead for box-predictor (e.g., heads with different self).
        This function setup the wrapper for the current head

        """

        self.model.roi_heads.box_predictor = head_class(
            self.model.roi_heads.box_predictor
        )

    def head_parameters(self):
        return self.model.roi_heads.box_head.parameters()

    def forward(self, inputs):
        # height = inputs[0]['height']
        # width = inputs[0]['width']
        # images = self.preprocess_image(inputs)
        # breakpoint()
        # if "instances" in inputs[0]:
        #     gt_instances = [x["instances"].to(self.device) for x in inputs]
        # else:
        #     gt_instances = None

        # features = self.model.backbone(images.tensor)
        # proposals, _ = self.model.proposal_generator(images, features, gt_instances)

        # instances, _ = self.model.roi_heads(images, features, proposals, gt_instances)
        # mask_features = [features[f] for f in self.model.roi_heads.in_features]
        # predictions_images = []

        # for i in range(len(instances)):
        #     predictions_images += [i] * len(instances[i])

        # predictions_boxes = [instances[i].pred_boxes for i in range(len(instances))]

        # pooled_features = self.model.roi_heads.box_pooler(
        #     mask_features, predictions_boxes
        # )
        # box_features = self.model.roi_heads.box_head(pooled_features)

        # predictions = self.postprocess(height, width, instances)

        # return predictions, box_features, predictions_images
        class_indices = [57, 58, 59, 60, 61, 62]
        all_results = []
        images = [inp['image_no_transformations'] for inp in inputs]
        inputs = self.processor(images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(outputs, target_sizes=[(1280, 1280) for i in range(len(images))], threshold=0.9, return_binary_maps=True)
        instances = []
        for result in results:
            idx_to_keep = []
            boxes = []
            logits = []
            scores = []
            classes_list = []
            pred_masks = []
            for idx, segment in enumerate(result['segments_info']):
                if segment['label_id'] in class_indices:
                    logit = torch.zeros(len(class_indices))
                    logit[segment['label_id'] - 57] = segment['score']
                    scores.append(segment['score'])
                    classes_list.append(segment['label_id'] - 57)
                    bb = cv2.boundingRect(result['segmentation'][idx].cpu().numpy().astype('uint8'))
                    x, y, w, h = bb
                    boxes.append(torch.tensor([x, y, x + w, y + h]).unsqueeze(0))
                    logits.append(logit.unsqueeze(0))
                    pred_masks.append(result['segmentation'][idx].bool())
            if len(classes_list) != 0:
                if len(logits) < 2:
                    gt_logits = logits[0]
                else:
                    gt_logits = torch.cat(logits, dim=0)
                instances.append({'instances':Instances(image_size = (1280, 1280), pred_boxes = Boxes(torch.cat(boxes, dim=0)), 
                                        gt_logits=gt_logits, scores = torch.Tensor(scores), 
                                        pred_classes = torch.Tensor(classes_list), pred_masks = torch.stack(pred_masks))})

            else:
                instances.append({'instances':Instances(image_size = (1280, 1280), pred_boxes = Boxes(torch.tensor(())), 
                                        gt_logits=torch.tensor(()), scores = torch.tensor(()), 
                                        pred_classes = torch.tensor(()), pred_masks = torch.tensor(()))})            
        

        return instances

    def __call__(self, inputs):
        with torch.no_grad():
            predictions = self.forward(inputs)
        return (predictions)

    def infer(self, original_image):
        """
                       Args:
                            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
        predictions (dict):
        the output of the model for one image only.
        See :doc:`/tutorials/models` for details about the format.
        """

        height, width = original_image[0].shape[:2]

        inputs = [
            {
                "image": torch.as_tensor(im).permute(2, 0, 1).to(self.device),
                "height": height,
                "width": width,
            }
            for im in original_image
        ]
        return self.__call__(inputs)

    def reinit_head(self, classes_idxs):
        self.model.roi_heads.num_classes = len(classes_idxs)

        if isinstance(self.model.roi_heads.box_predictor, MinimalPredictorWrapper):
            self.model.roi_heads.box_predictor.reinit_head(classes_idxs)
        else:
            classes_to_keep = np.array([*classes_idxs, 80])
            cls_bias = torch.nn.Parameter(
                self.model.roi_heads.box_predictor.cls_score.bias[classes_to_keep]
            )
            cls_weight = torch.nn.Parameter(
                self.model.roi_heads.box_predictor.cls_score.weight[classes_to_keep]
            )

            classes_to_keep = np.array([*classes_idxs])
            mask = np.repeat(classes_to_keep * 4, 4) + np.tile(
                np.arange(0, 4), len(classes_to_keep)
            )

            box_weight = torch.nn.Parameter(
                self.model.roi_heads.box_predictor.bbox_pred.weight[mask]
            )
            box_bias = torch.nn.Parameter(
                self.model.roi_heads.box_predictor.bbox_pred.bias[mask]
            )
            self.model.roi_heads.box_predictor.num_classes = len(classes_idxs)

            in_features = box_weight.shape[1]
            self.model.roi_heads.box_predictor.cls_score = nn.Linear(
                in_features, len(classes_to_keep) + 1
            )
            self.model.roi_heads.box_predictor.cls_score.bias = cls_bias
            self.model.roi_heads.box_predictor.cls_score.weight = cls_weight

            self.model.roi_heads.box_predictor.bbox_pred = nn.Linear(
                in_features, len(classes_idxs) * 4
            )
            self.model.roi_heads.box_predictor.bbox_pred.bias = box_bias
            self.model.roi_heads.box_predictor.bbox_pred.weight = box_weight

            if hasattr(self.model.roi_heads, "mask_head"):
                classes_to_keep = np.array([*classes_idxs])
                mask_weight = torch.nn.Parameter(
                    self.model.roi_heads.mask_head.predictor.weight[classes_to_keep]
                )
                mask_bias = torch.nn.Parameter(
                    self.model.roi_heads.mask_head.predictor.bias[classes_to_keep]
                )
                self.model.roi_heads.mask_head.predictor.weight = mask_weight
                self.model.roi_heads.mask_head.predictor.bias = mask_bias
                self.model.roi_heads.mask_head.predictor.num_classes = len(classes_idxs)

    @torch.no_grad()
    def head_forward(self, images, features, proposals):

        box_features = [features[f] for f in self.model.roi_heads.in_features]

        predictions_boxes = [x.proposal_boxes for x in proposals]

        pooled_features = self.model.roi_heads.box_pooler(
            box_features, predictions_boxes
        )
        box_features = self.model.roi_heads.box_head(pooled_features)

        predictions = self.model.roi_heads.box_predictor(box_features)
        pred_instances, _ = self.model.roi_heads.box_predictor.inference(
            predictions, proposals
        )

        outputs = self.model.roi_heads.forward_with_given_boxes(
            features, pred_instances
        )

        return outputs

    def postprocess(self, height, width, results):
        processed_results = []
        for results_per_image in results:
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.model.pixel_mean) / self.model.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.model.backbone.size_divisibility)
        return images


class HabitatPredictor(Predictor):
    def __init__(self, cfg, *args, **kwargs):
        cfg = setup_cfg(cfg).clone()

        labels = BBSense.CLASSES
        cfg['MODEL']['ROI_HEADS']['NUM_CLASSES'] = len(labels)
        cfg['MODEL']['ROI_BOX_HEAD']['BBOX_REG_LOSS_TYPE'] = 'giou'

        cfg.DATASETS.TEST = ["habitat_data set"]
        DatasetCatalog.register("habitat_dataset", lambda x: x)
        MetadataCatalog.get("habitat_dataset").thing_classes = list(labels.values())
        super().__init__(cfg, *args, **kwargs)


class MonteCarloPredictor(Predictor):
    def __init__(self, cfg, n_montecarlo_samples=10, *args, **kwargs):
        cfg = setup_cfg(cfg).clone()

        self.n_montecarlo_samples = n_montecarlo_samples
        super().__init__(cfg, *args, **kwargs)

    def reinit_head(self, classes_idxs):

        classes_to_keep = np.array([*classes_idxs, 80])
        self.model.roi_heads.box_predictor.cls_score[1].bias = torch.nn.Parameter(
            self.model.roi_heads.box_predictor.cls_score[1].bias[classes_to_keep]
        )
        self.model.roi_heads.box_predictor.cls_score[1].weight = torch.nn.Parameter(
            self.model.roi_heads.box_predictor.cls_score[1].weight[classes_to_keep]
        )
        classes_to_keep = np.array([*classes_idxs])
        mask = np.repeat(classes_to_keep * 4, 4) + np.tile(
            np.arange(0, 4), len(classes_to_keep)
        )
        self.model.roi_heads.box_predictor.bbox_pred.weight = torch.nn.Parameter(
            self.model.roi_heads.box_predictor.bbox_pred.weight[mask]
        )
        self.model.roi_heads.box_predictor.bbox_pred.bias = torch.nn.Parameter(
            self.model.roi_heads.box_predictor.bbox_pred.bias[mask]
        )
        self.model.roi_heads.box_predictor.num_classes = len(classes_idxs)
        self.model.roi_heads.num_classes = len(classes_idxs)

    def __call__(self, inputs):
        with torch.no_grad():
            images = self.preprocess_image(inputs)

            features = self.model.backbone(images.tensor)

            proposals, _ = self.model.proposal_generator(images, features, None)

            outs = [
                {'instances': x} for x in self.head_forward(images, features, proposals)
            ]
        return outs

    @torch.no_grad()
    def head_forward(self, images, features, proposals):
        """
        Monte-carlo roi_heads inference pass
        """

        box_features = [features[f] for f in self.model.roi_heads.in_features]

        predictions_boxes = [x.proposal_boxes for x in proposals]

        pooled_features = self.model.roi_heads.box_pooler(
            box_features, predictions_boxes
        )
        box_features = self.model.roi_heads.box_head(pooled_features)
        enable_dropout(self.model.roi_heads.box_predictor.cls_score)

        results = {i: [] for i in range(len(images))}
        boxes = None
        for _ in range(self.n_montecarlo_samples):
            predictions = self.model.roi_heads.box_predictor(box_features)
            boxes = self.model.roi_heads.box_predictor.predict_boxes(
                predictions, proposals
            )
            scores = self.model.roi_heads.box_predictor.predict_probs(
                predictions, proposals
            )
            for i in range(len(images)):
                results[i].append(scores[i].detach())
        mc_scores = tuple(
            [torch.stack([x for x in item]).mean(0) for item in results.values()]
        )
        shapes = [x.image_size for x in proposals]

        pred_instances, _ = fast_rcnn_inference(
            boxes,
            mc_scores,
            shapes,
            self.model.roi_heads.box_predictor.test_score_thresh,
            self.model.roi_heads.box_predictor.test_nms_thresh,
            self.model.roi_heads.box_predictor.test_topk_per_image,
        )

        outputs = self.model.roi_heads.forward_with_given_boxes(
            features, pred_instances
        )
        return outputs
