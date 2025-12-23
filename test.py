import numpy as np
import torch
import torchvision.transforms.v2 as T


def get_embeder(device="cpu"):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = model.to(device)
    model.eval()
    return model


def embed(image: np.ndarray, model: torch.nn.Module, device: str = "cpu") -> np.ndarray:
    transform = T.Compose(
        [
            T.Resize(size=(224, 224)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tensor_image = torch.from_numpy(image).permute(2, 0, 1)
    transformed = transform(tensor_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(transformed)

    embedding = logit.cpu().numpy()

    return embedding[0]


if __name__ == "__main__":
    model = get_embeder()
    image = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    print(image.shape)
    embedding = embed(image, model)
    print(embedding.shape)
