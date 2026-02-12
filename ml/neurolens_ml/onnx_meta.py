"""Model version and settings live inside each ONNX file, so every result can name the model that actually ran.

    python -m neurolens_ml.onnx_meta ml/models/segmenter3d.onnx version=seg-segresnet-v1 min_et_voxels=50 about="..."
    python -m neurolens_ml.onnx_meta ml/models/segmenter3d.onnx          # print what the file says

Keys: version (required), about (one line for the dashboard), min_et_voxels (segmenter: small-ET rule, tuned on val).
"""
import sys
from pathlib import Path

REQUIRED = ("version",)


def stamp(path: str | Path, **meta: str | int) -> None:
    """Write (or overwrite) metadata keys in an ONNX file, in place."""
    import onnx

    model = onnx.load(str(path))
    keep = {p.key: p.value for p in model.metadata_props} | {k: str(v) for k, v in meta.items()}
    missing = [k for k in REQUIRED if not keep.get(k)]
    if missing:
        raise ValueError(f"{Path(path).name}: missing {missing}")
    del model.metadata_props[:]
    for k, v in sorted(keep.items()):
        model.metadata_props.add(key=k, value=v)
    onnx.save(model, str(path))


def read(session) -> dict[str, str]:
    """Metadata of a loaded onnxruntime session; a file without a version is refused (it can't be traced)."""
    meta = dict(session.get_modelmeta().custom_metadata_map)
    if not meta.get("version"):
        raise ValueError("model file has no version: stamp it with neurolens_ml.onnx_meta")
    return meta


if __name__ == "__main__":
    path, pairs = sys.argv[1], sys.argv[2:]
    if pairs:
        stamp(path, **dict(p.split("=", 1) for p in pairs))
    import onnxruntime as ort

    print(read(ort.InferenceSession(path, providers=["CPUExecutionProvider"])))
