from torch import nn
from dataclasses import dataclass, field

from typing import List, Tuple, Dict, Any, Union, Literal, Optional
from torch import Tensor
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import os

# Import custom modules
from ...data import datatypes
from .image_encoder import Resnet18ImageEncoder, CLIPImageEncoder
from .text_encoder import HuggingFaceTextEncoder, CLIPTextEncoder
from ...utils.model_utils import freeze_model, mean_pooling, aggregate_embeddings
from transformers import AutoModel, AutoTokenizer, AutoProcessor


class ItemEncoder(nn.Module):
    def __init__(
        self, 
        model_name,
        enc_dim_per_modality,
        enc_norm_out,
        aggregation_method
    ):
        super().__init__()
        self.enc_dim_per_modality = enc_dim_per_modality
        self.aggregation_method = aggregation_method
        self.enc_norm_out = enc_norm_out
        self._build_encoders(model_name)
    
    def _build_encoders(self, model_name):
        self.image_enc = Resnet18ImageEncoder(
            d_embed=self.enc_dim_per_modality,
        )
        self.text_enc = HuggingFaceTextEncoder(
            embedding_size=self.enc_dim_per_modality,
            model_name_or_path=model_name
        )
        
    @property
    def d_embed(self):
        if self.aggregation_method == 'concat':
            d_model = self.enc_dim_per_modality * 2 
        else:
            d_model = self.enc_dim_per_modality
            
        return d_model
    
    @property
    def image_size(self):
        return self.image_enc.image_size

    def forward(self, images, texts, *args, **kwargs):        
        # Encode images and texts
        image_embeddings = self.image_enc(
            images, normalize=self.enc_norm_out, *args, **kwargs
        )
        text_embeddings = self.text_enc(
            texts, normalize=self.enc_norm_out, *args, **kwargs
        )
        # Aggregate embeddings
        encoder_outputs = aggregate_embeddings(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            aggregation_method=self.aggregation_method
        )
        
        return encoder_outputs


class ResNet18ImageOnlyEncoder(nn.Module):
    """Trainable ResNet-18 item encoder without a text modality."""

    def __init__(self, d_embed=64, enc_norm_out=True):
        super().__init__()
        self.enc_norm_out = enc_norm_out
        self.image_enc = Resnet18ImageEncoder(d_embed=d_embed, freeze=False)

    @property
    def d_embed(self):
        return self.image_enc.d_embed

    @property
    def image_size(self):
        return self.image_enc.image_size

    def forward(self, images, texts=None, *args, **kwargs):
        return self.image_enc(
            images, normalize=self.enc_norm_out, *args, **kwargs
        )

    
class CLIPItemEncoder(ItemEncoder):
    def __init__(
        self, 
        model_name,
        enc_norm_out,
        aggregation_method
    ):
        super().__init__(
            model_name=model_name,
            enc_dim_per_modality=512,
            enc_norm_out=enc_norm_out,
            aggregation_method=aggregation_method
        )
    
    def _build_encoders(self, model_name):
        self.image_enc = CLIPImageEncoder(
            model_name_or_path=model_name
        )
        self.text_enc = CLIPTextEncoder(
            model_name_or_path=model_name
        )
