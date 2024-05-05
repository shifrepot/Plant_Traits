from transformers import AutoImageProcessor, ViTModel
import torch
from torch import nn

class model4(nn.Module):
    def __init__(self):
        super(model4, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        self.regression_head = nn.Sequential(
            nn.Linear(768 + 27, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )


    def forward(self, image, data):
        inputs = self.image_processor(image, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs)
        pool = outputs.pooler_output
        concat = torch.cat([pool, data], dim=1)

        return self.regression_head(concat)