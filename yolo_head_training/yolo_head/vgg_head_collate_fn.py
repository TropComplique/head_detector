from typing import Tuple, List, Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from super_gradients.common.registry.registry import register_collate_function
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy

from .mesh_sample import MeshEstimationSample

__all__ = ["VGGHeadCollateFN", "undo_flat_collate_tensors_with_batch_index", "flat_collate_tensors_with_batch_index"]


@register_collate_function()
class VGGHeadCollateFN:
    def __init__(self, set_image_to_none: bool = True):
        """

        :param set_image_to_none: If True, image and mask properties for each sample will be set to None after collation.
                                  After we collate images from samples into batch individual images are not needed anymore.
                                  Keeping them in sample slows down data transfer time and slows training 2X.
                                  If True, image and mask properties will be set to None after collation.
                                  If False, image and mask properties will be converted to torch tensors and kept in the sample.
        """
        self.set_image_to_none = set_image_to_none

    def __call__(self, batch: List[MeshEstimationSample]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor], Dict]:
        """
        Collate samples into a batch.
        This collate function is compatible with VGGHead model

        :param batch: A list of samples from the dataset. Each sample is a tuple of (image, (boxes, joints), extras)
        :return: Tuple of (images, (boxes, joints), extras)
        - images: [Batch, 3, H, W]
        - boxes: [NumInstances, 5], last dimension represents (batch_index, x1, y1, x2, y2) of all boxes in a batch
        - joints: [NumInstances, NumJoints, 4] of all poses in a batch. Last dimension represents (batch_index, x, y, visibility)
        - extras: A dict of extra information per image need for metric computation
        """
        all_images = []
        all_boxes = []
        all_joints = []
        all_vertices = []
        all_rots = []

        for sample in batch:
            # Generate targets
            boxes, vertices_2d, vertices_3d, rotations = self._get_targets(sample)
            all_boxes.append(boxes)
            all_joints.append(vertices_2d)
            all_vertices.append(vertices_3d)
            all_rots.append(rotations)

            # Convert image & mask to tensors
            # Change image layout from HWC to CHW
            sample.image = torch.from_numpy(np.transpose(sample.image, [2, 0, 1]))
            all_images.append(sample.image)

            # Remove image and mask from sample because at this point we don't need them anymore
            if self.set_image_to_none:
                sample.image = None

            # Make sure additional samples are None, so they don't get collated as it causes collate to slow down
            sample.additional_samples = None
        all_images = default_collate(all_images)
        boxes = flat_collate_tensors_with_batch_index(all_boxes)
        joints = flat_collate_tensors_with_batch_index(all_joints)
        vertices = flat_collate_tensors_with_batch_index(all_vertices)
        rotations = flat_collate_tensors_with_batch_index(all_rots)
        extras = {"gt_samples": batch}
        return all_images, (boxes, joints, vertices, rotations), extras

    def _get_targets(self, sample: MeshEstimationSample) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Generate targets for training VGGHead from a single MeshEstimationSample
        :param sample: Input PoseEstimationSample
        :return:       Tuple of (boxes, joints, is_crowd)
                       - boxes - [NumInstances, 4] - torch tensor of bounding boxe (XYXY) for each mesh instance in a sample
                       - vertices_2d - [NumInstances, NumJoints, 3] - torch tensor of pose joints for each mesh instance in a sample
                       - vertices_3d - [NumInstances, NumVertices, 3] - torch tensor of 3D vertices for each mesh instance in a sample
                       - rotations - [NumInstances, 3, 3] - torch tensor of rotation matrices for each mesh instance in a sample
        """
        boxes_xyxy = xywh_to_xyxy(sample.bboxes_xywh, image_shape=None)

        return torch.from_numpy(boxes_xyxy), \
               torch.from_numpy(sample.vertices_2d), \
               torch.from_numpy(sample.vertices_3d),\
               torch.from_numpy(sample.rotation_matrix),


def flat_collate_tensors_with_batch_index(labels_batch: List[Tensor]) -> Tensor:
    """
    Concatenate tensors along the first dimension and add a sample index as the first element in the last dimension.

    :param labels_batch: A list of targets per image (each of arbitrary length: [N1, ..., C], [N2, ..., C], [N3, ..., C],...)
    :return:             A single tensor of shape [N1+N2+N3+..., ..., C+1].
    """
    labels_batch_indexed = []
    for i, labels in enumerate(labels_batch):
        batch_column = labels.new_ones(labels.shape[:-1] + (1,)) * i
        labels = torch.cat((batch_column, labels), dim=-1)
        labels_batch_indexed.append(labels)
    return torch.cat(labels_batch_indexed, 0)


def undo_flat_collate_tensors_with_batch_index(flat_tensor: Tensor, batch_size: int) -> List[Tensor]:
    """
    Unrolls the flat tensor into list of tensors per batch item.
    As name suggest it undoes what flat_collate_tensors_with_batch_index does.

    :param flat_tensor: Tensor of shape [N1+N2+N3+..., ..., C+1].
    :param batch_size:  The batch size (Number of items in the batch)
    :return:            List of tensors [N1, ..., C], [N2, ..., C], [N3, ..., C],...
    """
    items = []
    batch_index_roi = [slice(None)] + [0] * (flat_tensor.ndim - 1)
    batch_index = flat_tensor[batch_index_roi]
    for i in range(batch_size):
        mask = batch_index == i
        items.append(flat_tensor[mask][..., 1:])
    return items
