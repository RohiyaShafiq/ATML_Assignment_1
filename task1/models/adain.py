import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


ADAIN_REPO = Path("/content/pytorch-AdaIN")

if str(ADAIN_REPO) not in sys.path:
    sys.path.append(str(ADAIN_REPO))

import net
from function import adaptive_instance_normalization


# AdaIN Style Transfer
class AdaINStyleTransfer:

    def __init__(
        self,
        device,
        vgg_path=None,
        decoder_path=None,
    ):

        self.device = device

        if vgg_path is None:
            vgg_path = ADAIN_REPO / "models" / "vgg_normalized.pth"

        if decoder_path is None:
            decoder_path = ADAIN_REPO / "models" / "decoder.pth"

        # Load VGG encoder
        self.vgg = net.vgg

        self.vgg.load_state_dict(
            torch.load(
                vgg_path,
                map_location=device))

        # AdaIN uses VGG up to relu4_1
        self.vgg = nn.Sequential(
            *list(self.vgg.children())[:31])

        # Load decoder
        self.decoder = net.decoder

        self.decoder.load_state_dict(
            torch.load(
                decoder_path,
                map_location=device))

        self.vgg = self.vgg.to(device)
        self.decoder = self.decoder.to(device)

        self.vgg.eval()
        self.decoder.eval()

        for parameter in self.vgg.parameters():
            parameter.requires_grad = False

        for parameter in self.decoder.parameters():
            parameter.requires_grad = False

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

    # Transfer style
    @torch.no_grad()
    def transfer(
        self,
        content_image,
        style_image,
        alpha=0.8):

        content = self.transform(
            content_image.convert("RGB")
        ).unsqueeze(0).to(self.device)

        style = self.transform(
            style_image.convert("RGB")
        ).unsqueeze(0).to(self.device)

        # Encode
        content_features = self.vgg(content)
        style_features = self.vgg(style)

        # AdaIN
        target_features = adaptive_instance_normalization(
            content_features,
            style_features)

        # Control style strength
        target_features = (
            alpha * target_features
            + (1 - alpha) * content_features)

        # Decode
        generated = self.decoder(target_features)
        generated = generated.clamp(0, 1)

        # Convert back to PIL
        generated = generated.squeeze(0).cpu()
        generated = transforms.ToPILImage()(generated)

        return generated