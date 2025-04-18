from typing import List, Union, Tuple, Dict, Any

import cv2
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from head_detector.head_info import HeadMetadata, Bbox, FlameParams
from head_detector.detection_result import PredictionResult
from head_detector.utils import nms, calculate_rpy
from head_detector.flame import FLAMELayer, reproject_spatial_vertices
from head_detector.pncc_processor import PNCCProcessor


REPO_ID = "okupyn/vgg_heads"


class HeadDetector:
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32,
                 model: str = "vgg_heads_l", image_size: int = 640,
                 weights_path: str | None = None, use_compile: bool = False):
        self._image_size = image_size
        self._device = device
        self._dtype = dtype
        self._flame = FLAMELayer().to(self._device)
        self.model = self._read_model(model, weights_path)
        self.pncc_processor = PNCCProcessor()

        if use_compile:
            torch.set_float32_matmul_precision("high")
            compile_arguments = {"backend": "inductor", "mode": "default", "fullgraph": False, "dynamic": False}
            self._flame = torch.compile(self._flame, **compile_arguments)

    def _read_model(self, model: str, weights_path: str) -> torch.jit.ScriptModule:
        model_path = hf_hub_download(REPO_ID, f"{model}.trcd") if weights_path is None else weights_path
        loaded_model = torch.jit.load(model_path)
        loaded_model.to(self._dtype).to(self._device)
        loaded_model.eval()
        return loaded_model

    def _transform_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int], float]:
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = self._image_size, int(w * self._image_size / h)
        else:
            new_h, new_w = int(h * self._image_size / w), self._image_size
        scale = self._image_size / max(image.shape[:2])
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        pad_w = self._image_size - image.shape[1]
        pad_h = self._image_size - image.shape[0]
        image = cv2.copyMakeBorder(image, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2, cv2.BORDER_CONSTANT, value=127)
        image_input = torch.from_numpy(image).to(self._device).permute(2, 0, 1).unsqueeze(0).to(self._dtype) / 255.0
        return image_input, (pad_w // 2, pad_h // 2), scale

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        image, padding, scale = self._transform_image(image)
        return image, {"padding": padding, "scale": scale}

    def _process(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            output = self.model(image)
        return output

    def _parse_predictions(self, bboxes_xyxy: torch.Tensor, scores: torch.Tensor, flame_params: torch.Tensor, cache: Dict[str, Any]):
        padding = cache["padding"]
        scale = cache["scale"]
        bboxes_xyxy = bboxes_xyxy.cpu().numpy()
        scores = scores.cpu().numpy()
        _, _, final_3d_pts = reproject_spatial_vertices(self._flame, flame_params, to_2d=False)
        final_3d_pts[:, :, 0] -= padding[0]
        final_3d_pts[:, :, 1] -= padding[1]
        final_3d_pts = (final_3d_pts / scale).cpu().numpy()
        bboxes_xyxy = bboxes_xyxy.clip(0, self._image_size)
        bboxes_xyxy[:, [0, 2]] -= padding[0]
        bboxes_xyxy[:, [1, 3]] -= padding[1]
        bboxes_xyxy /= scale
        bboxes_xyxy = np.rint(bboxes_xyxy).astype(int)
        result = []
        flame_params = flame_params.detach().cpu()
        for bbox, score, vertices in zip(bboxes_xyxy, scores, final_3d_pts):
            box = Bbox(x=bbox[0], y=bbox[1], w=bbox[2] - bbox[0], h=bbox[3] - bbox[1])
            result.append(
                HeadMetadata(
                    bbox=box,
                    score=score,
                    flame_params=None,
                    vertices_3d=vertices,
                    head_pose=None,
                )
            )
        return result

    def _postprocess(self, predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], cache: Dict[str, Any], confidence_threshold: float) -> List[HeadMetadata]:
        boxes, scores, flame_params = predictions  # they have shapes [1, 1000, 4], [1, 1000, 1], and [1, 1000, 413]
        boxes, scores, flame_params = nms(boxes, scores, flame_params, confidence_threshold=confidence_threshold)
        return self._parse_predictions(boxes, scores, flame_params, cache)

    def __call__(self, image: np.ndarray, confidence_threshold: float = 0.5) -> tuple:
        """
        The input tensor represents an RGB image
        with values in the range [0, 255].

        Arguments:
            image: a numpy uint8 array with shape [h, w, 3].
        Returns:
            pncc: a numpy uint8 array with shape [h, w, 3].
            heads: a list of HeadMetadata.
        """
        original_image = image
        image, cache = self._preprocess(original_image)
        predictions = self._process(image)
        heads = self._postprocess(predictions, cache, confidence_threshold)
        return self.pncc_processor(original_image, heads), heads
