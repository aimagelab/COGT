import torch
from torch import nn
from models import open_clip_lora
import logging as python_logging

python_logging.basicConfig(level=python_logging.INFO)


class CLIPWrapper(nn.Module):
    def __init__(self,
                 backbone: str,
                 pretrained_path: str,
                 device: str):
        """
        :param backbone:
        :param pretrained_path
        :param device:
        """

        super().__init__()

        self.device = device

        # open_clip case
        self.model, _, self.image_preprocess = open_clip_lora.create_model_and_transforms(backbone,
                                                                                          device=self.device,
                                                                                          pretrained=pretrained_path)

        self.tokenizer = {"name": "open_clip", "tokenizer": open_clip_lora.get_tokenizer(backbone)}

        self.model = self.model.to(device)
        self.model = self.model.float()
        self.print_trainable_parameters()

    def get_image_embeds(self,
                         image_embeds: torch.Tensor,
                         normalize: bool = True) -> torch.Tensor:
        """
        Encode image_embeds according to the model below

        :param image_embeds:
        :param normalize:
        :return:
        """

        image_embeds = self.model.encode_image(image_embeds, normalize=normalize)

        return image_embeds

