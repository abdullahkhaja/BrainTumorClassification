# model/predictor.py
import torch
from torchvision import transforms, models
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# same transform
_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# load your locally‑trained checkpoint
ckpt = torch.load(
    os.path.join("model","checkpoints","brain_tumor_model.pth"),
    map_location=device
)
num_classes = ckpt["num_classes"]
class_names = ckpt["class_names"]

# rebuild model architecture
_model = models.resnet18(weights=None)
_model.fc = torch.nn.Linear(_model.fc.in_features, num_classes)
_model.load_state_dict(ckpt["model_state_dict"])
_model.to(device).eval()

def predict(image_path: str):
    img = Image.open(image_path).convert("RGB")
    x = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = _model(x)
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())
    return {
        "class":      class_names[idx],
        "confidence": float(probs[idx])
    }
