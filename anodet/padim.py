"""
Provides classes and functions for working with PaDiM.
"""

import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as T
from tqdm import tqdm
from typing import Optional, Callable, List, Tuple
from .feature_extraction import ResnetEmbeddingsExtractor
from .utils import pytorch_cov, mahalanobis, split_tensor_and_run_function
from collections import OrderedDict


class Padim(torch.nn.Module):
    """A padim model with functions to train and perform inference."""

    def __init__(self, backbone: str = 'resnet18',
                 device: torch.device = torch.device('cpu'),
                 mean: Optional[torch.Tensor] = None,
                 cov_inv: Optional[torch.Tensor] = None,
                 channel_indices: Optional[torch.Tensor] = None,
                 layer_indices: Optional[List[int]] = None,
                 layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 feat_dim: Optional[int] = 50) -> None:

        """Construct the model and initialize the attributes

        Args:
            backbone: The name of the desired backbone. Must be one of: [resnet18, wide_resnet50].
            device: The device where to run the model.
            mean: A tensor with the mean vectors of each patch with size (D, H, W), \
                where D is the number of channel_indices.
            cov_inv: A tensor with the inverse of the covariance matrices of each patch \
                with size (D, D, H, W), where D is the number of channel_indices.
            channel_indices: A tensor with the desired channel indices to extract \
                from the backbone, with size (D).
            layer_indices: A list with the desired layers to extract from the backbone, \
            allowed indices are 1, 2, 3 and 4.
            layer_hook: A function that can modify the layers during extraction.
        """

        super(Padim, self).__init__()
        
        self.device = device
        # Register as a submodule for proper ONNX export
        self.embeddings_extractor = ResnetEmbeddingsExtractor(backbone, self.device)
        
        # Initialize buffers - register_buffer handles None values properly
        self.register_buffer('_mean', mean)
        self.register_buffer('_cov_inv', cov_inv)

        self.layer_indices = layer_indices
        if self.layer_indices is None:
            self.layer_indices = [0]

        self.layer_hook = layer_hook
        self.to_device(self.device)
        # Register channel_indices as a buffer for ONNX compatibility
        if channel_indices is not None:
            self.register_buffer('channel_indices', channel_indices)
        else:
            if backbone == 'resnet18':
                self.net_feature_size = OrderedDict(
                    [(0, [64]), (1, [128]), (2, [256]), (3, [512])])
                
                
            elif backbone == 'wide_resnet50':
                self.net_feature_size = OrderedDict(
                    [(0, [255]), (1, [512]), (2, [1024]), (3, [2048])]
                )                
                                            
            self.register_buffer(
                "channel_indices",
                get_dims_indices(self.layer_indices, feat_dim, self.net_feature_size),
            )

    @property
    def mean(self):
        """Get the mean tensor."""
        return self._mean
    
    @property
    def cov_inv(self):
        """Get the inverse covariance tensor."""
        return self._cov_inv

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for ONNX compatibility.

        Args:
            x: A batch of input images, with dimension (B, C, H, W).

        Returns:
            image_scores: A tensor with the image level scores, with dimension (B).
            score_map: A tensor with the patch level scores, with dimension (B, H, W)
        """
        # Extract features using the embeddings extractor
        embedding_vectors = self.embeddings_extractor(x,
                                                      channel_indices=self.channel_indices,
                                                      layer_hook=self.layer_hook,
                                                      layer_indices=self.layer_indices
                                                      )

        # Calculate Mahalanobis distance
        patch_scores = mahalanobis(self.mean, self.cov_inv, embedding_vectors)

        # Reshape to square patches - use a more ONNX-friendly approach
        batch_size = x.shape[0]
        num_patches = embedding_vectors.shape[1]
        # patch_width = int(torch.sqrt(torch.tensor(num_patches, dtype=torch.float32)).item())
        patch_width = int(torch.sqrt(num_patches.float()).item())

        patch_scores = patch_scores.view(batch_size, patch_width, patch_width)

        # Interpolate to original image size
        score_map = F.interpolate(patch_scores.unsqueeze(1), 
                                  size=(x.shape[2], x.shape[3]),
                                  mode='bilinear', 
                                  align_corners=False)
        
        # Remove the channel dimension
        score_map = score_map.squeeze(1)

        # Apply gaussian blur - create the blur operation inline for ONNX
        # Using a simpler approach that's more ONNX-friendly
        score_map = T.GaussianBlur(33, sigma=4)(score_map)     

        # Calculate image-level scores
        image_scores = torch.max(score_map.view(batch_size, -1), dim=1)[0]

        return image_scores, score_map

    def export_onnx(self, filepath: str, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> None:
        """Export the model to ONNX format.

        Args:
            filepath: Path where to save the ONNX model.
            input_shape: Shape of the input tensor (B, C, H, W).
        """
        assert self.mean is not None and self.cov_inv is not None, \
            "The model must be trained before exporting to ONNX"
        
        self.eval()  # Set to evaluation mode
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape, device=self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['image_scores', 'score_map'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'image_scores': {0: 'batch_size'},
                'score_map': {0: 'batch_size'}
            },
            verbose=True
        )

    def to_device(self, device: torch.device) -> None:
        """Perform device conversion on backone, mean, cov_inv and channel_indices

        Args:
            device: The device where to run the model.

        """

        self.device = device
        if self.embeddings_extractor is not None:
            self.embeddings_extractor.to_device(device)
        # Buffers are automatically moved with the module, so no need to manually move them

    def fit(self, dataloader: torch.utils.data.DataLoader, extractions: int = 1) -> None:
        """Fit the model (i.e. mean and cov_inv) to data.

        Args:
            dataloader: A pytorch dataloader, with sample dimensions (B, D, H, W), \
                containing normal images.
            extractions: Number of extractions from dataloader. Could be of interest \
                when applying random augmentations.

        """
        embedding_vectors = None
        for i in range(extractions):
            extracted_embedding_vectors = self.embeddings_extractor.from_dataloader(
                dataloader,
                channel_indices=self.channel_indices,
                layer_hook=self.layer_hook,
                layer_indices=self.layer_indices
            )
            if embedding_vectors is None:
                embedding_vectors = extracted_embedding_vectors
            else:
                embedding_vectors = torch.cat((embedding_vectors, extracted_embedding_vectors), 0)

        mean = torch.mean(embedding_vectors, dim=0)
        cov = pytorch_cov(embedding_vectors.permute(1, 0, 2), rowvar=False) \
            + 0.01 * torch.eye(embedding_vectors.shape[2])
        # Run inverse function on splitted tensor to save ram memory
        cov_inv = split_tensor_and_run_function(func=torch.inverse,
                                               tensor=cov,
                                               split_size=1)
        
        # Register as buffers for proper model state management
        self.register_buffer('_mean', mean)
        self.register_buffer('_cov_inv', cov_inv)

    def predict(self, batch: torch.Tensor, gaussian_blur: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make a prediction on test images.

        Args:
            batch: A batch of test images, with dimension (B, D, h, w).
            gaussian_blur: Whether to apply gaussian blur to score maps.

        Returns:
            image_scores: A tensor with the image level scores, with dimension (B).
            score_map: A tensor with the patch level scores, with dimension (B, H, W)

        """

        assert self.mean is not None and self.cov_inv is not None, \
            "The model must be trained or provided with mean and cov_inv"

        embedding_vectors = self.embeddings_extractor(batch,
                                                      channel_indices=self.channel_indices,
                                                      layer_hook=self.layer_hook,
                                                      layer_indices=self.layer_indices
                                                      )

        patch_scores = mahalanobis(self.mean, self.cov_inv, embedding_vectors)

        patch_width = int(math.sqrt(embedding_vectors.shape[1]))
        patch_scores = patch_scores.reshape(batch.shape[0], patch_width, patch_width)

        score_map = F.interpolate(patch_scores.unsqueeze(1), size=batch.shape[2],
                                  mode='bilinear', align_corners=False).squeeze()


        if batch.shape[0] == 1:
            score_map = score_map.unsqueeze(0)

        if gaussian_blur:
            score_map = T.GaussianBlur(33, sigma=4)(score_map)

        image_scores = torch.max(score_map.reshape(score_map.shape[0], -1), -1).values

        return image_scores, score_map

    def evaluate(self, dataloader: torch.utils.data.DataLoader) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        """Run predict on all images in a dataloader and return the results.

        Args:
            dataloader: A pytorch dataloader, with sample dimensions (B, D, H, W), \
                containing normal images.

        Returns:
            images: An array containing all input images.
            image_classifications_target: An array containing the target \
                classifications on image level.
            masks_target: An array containing the target classifications on patch level.
            image_scores: An array containing the predicted scores on image level.
            score_maps: An array containing the predicted scores on patch level.

        """

        images = []
        image_classifications_target = []
        masks_target = []
        image_scores = []
        score_maps = []

        for (batch, image_classifications, masks) in tqdm(dataloader, 'Inference'):
            batch_image_scores, batch_score_maps = self.predict(batch)

            images.extend(batch.cpu().numpy())
            image_classifications_target.extend(image_classifications.cpu().numpy())
            masks_target.extend(masks.cpu().numpy())
            image_scores.extend(batch_image_scores.cpu().numpy())
            score_maps.extend(batch_score_maps.cpu().numpy())

        return np.array(images), np.array(image_classifications_target), \
            np.array(masks_target).flatten().astype(np.uint8), \
            np.array(image_scores), np.array(score_maps).flatten()


def get_dims_indices(layers, feature_dim, net_feature_size):
    random.seed(1024)
    torch.manual_seed(1024)

    total = 0
    for layer in layers:
        total += net_feature_size[layer][0]
    feature_dim = min(feature_dim, total)

    return torch.tensor(random.sample(range(0, total), feature_dim))
