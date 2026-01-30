from __future__ import annotations

import os
import sys
from pathlib import Path

import ray
from ray import serve
from ray.serve.config import gRPCOptions


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _generated_dir() -> Path:
    return Path(__file__).resolve().parent / "generated"


def _ensure_import_paths() -> None:
    # Ensure:
    #  - `triton_compat` package is importable.
    #  - Generated *_pb2.py modules are importable as top-level modules
    #    because grpcio-tools generates absolute imports like `import model_config_pb2`.
    root = str(_repo_root())
    gen = str(_generated_dir())

    for p in (gen, root):
        if p not in sys.path:
            sys.path.insert(0, p)

    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    for p in (gen, root):
        if p not in parts:
            parts.insert(0, p)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def main() -> None:
    _ensure_import_paths()

    ray.init(
        runtime_env={"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}}
    )

    serve.start(
        grpc_options=gRPCOptions(
            port=8001,
            grpc_servicer_functions=[
                # Import paths for generated add_* functions (strings).
                "grpc_service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server",
                "health_pb2_grpc.add_HealthServicer_to_server",
            ],
        ),
    )

    from triton_compat.app import app

    # Exactly one application; explicit route_prefix required by Serve.
    serve.run(app, name="triton", route_prefix="/", blocking=True)


if __name__ == "__main__":
    main()
