import typing
import torch
import open_clip


class TextEncoder:
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        if model_name == "maskclip":
            from featup.featurizers.maskclip.clip import tokenize
            self.model = typing.cast(torch.nn.Module, torch.hub.load("mhamilton723/FeatUp", "maskclip", use_norm=False).model.model)
            self.model.to(device)
            self.model.eval()
            self.tokenizer = tokenize
        elif model_name == "SAM2OpenCLIP":
            self.model, _, _ = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
            self.model.to(device)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')
        elif model_name == "SAMOpenCLIP":
            self.model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
            self.model.to(device)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
        else:
            raise ValueError(f"Model {model_name} not supported")
    
    def encode_text(self, text: str):
        if self.model_name == "maskclip":
            return self.model.encode_text(self.tokenizer(text).cuda())
        elif self.model_name == "SAM2OpenCLIP":
            return self.model.encode_text(self.tokenizer(text).cuda())[:, :512]
        elif self.model_name == "SAMOpenCLIP":
            return self.model.encode_text(self.tokenizer(text).cuda())[:, :512]
        else:
            raise ValueError(f"Model {self.model_name} not supported")
