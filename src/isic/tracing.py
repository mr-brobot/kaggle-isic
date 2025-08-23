from typing import Any, Dict, Union

import mlflow
import pandas as pd
import torch
from PIL import Image


def summarize_tensor(obj: Any) -> Union[Dict[str, Any], Any]:
    """Convert PyTorch tensor to shape/dtype summary"""
    if isinstance(obj, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }
    return obj


def summarize_dataframe(obj: Any) -> Union[Dict[str, Any], Any]:
    """Convert dataframe to shape/columns summary"""
    if isinstance(obj, pd.DataFrame):
        return {"type": "dataframe", "shape": obj.shape, "columns": list(obj.columns)}
    return obj


def summarize_image(obj: Any) -> Union[Dict[str, Any], Any]:
    """Convert PIL image to size/mode summary"""
    if isinstance(obj, Image.Image):
        return {"type": "image", "size": obj.size, "mode": obj.mode}
    return obj


def tensor_filter(span: mlflow.entities.LiveSpan) -> None:
    """Replace tensors with summaries"""
    if span.inputs:
        span.set_inputs({k: summarize_tensor(v) for k, v in span.inputs.items()})

    if span.outputs and isinstance(span.outputs, torch.Tensor):
        span.set_outputs(summarize_tensor(span.outputs))


def dataframe_filter(span: mlflow.entities.LiveSpan) -> None:
    """Replace dataframes with summaries"""
    if span.inputs:
        span.set_inputs({k: summarize_dataframe(v) for k, v in span.inputs.items()})

    if span.outputs and isinstance(span.outputs, pd.DataFrame):
        span.set_outputs(summarize_dataframe(span.outputs))


def image_filter(span: mlflow.entities.LiveSpan) -> None:
    """Replace images with summaries"""
    if span.inputs:
        span.set_inputs({k: summarize_image(v) for k, v in span.inputs.items()})

    if span.outputs and isinstance(span.outputs, Image.Image):
        span.set_outputs(summarize_image(span.outputs))


# Apply all filters to your tracing


# Usage example
@mlflow.trace
def process_data(
    image: Image.Image, df: pd.DataFrame, tensor: torch.Tensor
) -> Dict[str, Any]:
    # Your processing logic here
    processed_tensor = tensor * 2
    filtered_df = df.head(100)
    resized_image = image.resize((224, 224))
    return {"tensor": processed_tensor, "df": filtered_df, "image": resized_image}
