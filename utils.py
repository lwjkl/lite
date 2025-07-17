import numpy
import torch
import torchvision
import torchvision.transforms.v2 as T


class ResNet50FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        self.resnet = torch.nn.Sequential(
            *list(
                torchvision.models.resnet50(
                    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                ).children()
            )[:-1]
        )

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        return x


def get_embeder(device="cpu"):
    model = ResNet50FeatureExtractor()
    model = model.to(device)
    model.eval()
    return model


def embed(
    image: numpy.ndarray, model: torch.nn.Module, device: str = "cpu"
) -> numpy.ndarray:
    transform = T.Compose(
        [
            T.Resize(size=(224, 224)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    image = torch.from_numpy(image).permute(2, 0, 1)
    transformed = transform(image).unsqueeze(0).to(device)

    # print(transformed.shape)

    # if "cuda" in device:
    #     model.to(device)

    with torch.no_grad():
        logit = model(transformed)

    # print(logit.shape)
    embedding = logit.cpu().numpy()

    return embedding[0]
