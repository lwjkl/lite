import numpy
import torch
import torchvision.transforms.v2 as T


def embed(
    image: numpy.ndarray, model: torch.nn.Module, device: str = "cpu"
) -> numpy.ndarray:
    transform = T.Compose(
        [
            T.Resize(size=(224, 224)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    image = torch.from_numpy(image).permute(2, 0, 1)
    transformed = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(transformed)

    embedding = logit.cpu().numpy()

    return embedding[0]


def get_embeder(device="cpu"):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(device)
    model.eval()
    return model
