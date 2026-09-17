"""
gradcam.py
ConvNeXt 계열 모델 GradCAM 생성 + 오버레이
"""
import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self._activations = None
        self._gradients   = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        input_tensor : (1, 3, H, W), requires_grad=True
        반환         : (H, W) 0~1 정규화 히트맵
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        weights = self._gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().cpu().numpy()

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam  = cv2.resize(cam, (w, h))
        cam  = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGB uint8 이미지에 JET 컬러맵 히트맵 오버레이. 반환도 RGB."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)
