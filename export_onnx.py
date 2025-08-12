# export_onnx.py
import torch
import onnx
from model.predictor import _model  # your built model from predictor.py

# create a dummy input with the same shape your model expects
dummy = torch.randn(1, 3, 224, 224, device="cpu")

# export
torch.onnx.export(
    _model.cpu(),             # ensure on CPU
    dummy,
    "static/model.onnx",      # target in your static/ folder
    input_names=["image"],
    output_names=["logits"],
    opset_version=13,
    dynamic_axes={"image": {0: "batch"}}
)

print("✅ model.onnx exported to static/model.onnx")
