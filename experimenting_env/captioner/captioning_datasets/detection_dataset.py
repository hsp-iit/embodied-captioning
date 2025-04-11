import os
import cv2
import numpy as np
import re

from torch.utils.data import Dataset as BaseDataset, DataLoader
from tqdm import tqdm


def expand_bounding_box(bb, img_size, margin=0):
    bbox_temp = [bb[0] - margin if (bb[0] - margin) >= 0 else 0,
                 bb[1] - margin if (bb[1] - margin) >= 0 else bb[1],
                 bb[2] + margin if (bb[2] + margin) <= img_size.size[0] else bb[2],
                 bb[3] + margin if (bb[3] + margin) <= img_size.size[1] else bb[3]]
    return bbox_temp


class DetectionDataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)

    """

    def __init__(
            self,
            data_dir
    ):
        self.ids_img = os.listdir(data_dir)
        self.images_fps = []
        self.bboxes_fps = []
        for image_id in self.ids_img:
            if image_id.__contains__("color_sensor") or image_id.__contains__("rgb"):
                self.images_fps.append(os.path.join(data_dir, image_id))
            elif image_id.__contains__("bbsgt"):
                self.bboxes_fps.append(os.path.join(data_dir, image_id))
        self.images_fps.sort()
        self.bboxes_fps.sort()

    def __getitem__(self, i):
        filename = os.path.basename(self.images_fps[i])

        ids2names = {}
        ids2classes = {}
        ids2bbs = {}
        # ids2masks = {}
        episode_id = int(re.search(r'episode_(\d+)_', filename).group(1))
        idx = -1

        # Read data
        rgb = np.load(self.images_fps[i])
        bbsgt = np.load(self.bboxes_fps[i], allow_pickle=True).item()['instances']
        if len(bbsgt.infos) != 0:
            # for info, cls, bb, mask in zip(
            #         bbsgt.infos,
            #         bbsgt.pred_classes,
            #         bbsgt.pred_boxes,
            #         bbsgt.pred_masks,
            # ):
            for info, cls, bb in zip(
                    bbsgt.infos,
                    bbsgt.pred_classes,
                    bbsgt.pred_boxes,
            ):

                if 'id_object' in info.keys():
                    idx = info['id_object']
                elif 'object_id' in info.keys():
                    idx = info['object_id']
                key = (episode_id, idx)
                if key in ids2names:
                    ids2names[key].append(filename)
                else:
                    ids2names[key] = [filename]

                bboxes = bb.tolist()
                if key in ids2bbs:
                    ids2bbs[key].append(bb)
                else:
                    ids2bbs[key] = [bb]

                ids2classes[key] = cls

        sample_final = {'rgb': rgb.copy(), 'bboxes': ids2bbs, 'cls': ids2classes, 'filename': filename,
                        "episode_id": episode_id, "object_id": idx}

        return sample_final

    def __len__(self):
        return len(self.images_fps)


if __name__ == '__main__':
    DATA_DIR = "/media/tapicella/Win11_OS/Users/tapicella/Downloads/gibson_dataset/gibson_finetuning"
    dataset = DetectionDataset(DATA_DIR)
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Visualise some samples
    for i, sample_batch in enumerate(tqdm(dataset_loader)):
        # Load filename
        filename = sample_batch['filename'][0]

        # Load image
        image = sample_batch['rgb'].cpu().detach().numpy()[0].astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Load bounding box
        bboxes = list(sample_batch['bboxes'].values())[0][0].cpu().detach().numpy()[0].astype(np.int)

        cv2.imshow("RGB", image)

        # Crop image
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bboxes
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            image_crop = image[ymin:ymax, xmin:xmax, :]
            cv2.imshow("Crop", image_crop)
            overlay_vis = image.copy()
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            color = (0, 0, 255)
            thickness = 2
            overlay_vis = cv2.rectangle(overlay_vis, start_point, end_point, color, thickness)
            cv2.imshow("Bounding box", overlay_vis)
            cv2.waitKey(0)
