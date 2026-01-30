# Triton-compatible gRPC over Ray Serve (MVP)

This package implements a minimal subset of NVIDIA Triton Inference Server’s gRPC protocol on top of Ray Serve’s native gRPC proxy.

## Model

- Model name: `simple`
- Inputs:
  - `INPUT0`: `INT32`, shape `[1,16]`
  - `INPUT1`: `INT32`, shape `[1,16]`
- Outputs:
  - `OUTPUT0`: `INT32`, shape `[1,16]` = `INPUT0 + INPUT1`
  - `OUTPUT1`: `INT32`, shape `[1,16]` = `INPUT0 - INPUT1`

## Supported Triton RPCs (MVP)

Implemented from `grpc_service.proto`:

- `ServerLive`
- `ServerReady`
- `ModelReady`
- `ServerMetadata` (`name` is `"triton"`)
- `ModelMetadata`
- `ModelConfig`
- `ModelInfer` (unary; raw tensor payloads via `raw_input_contents` / `raw_output_contents`)

All other unary RPCs return gRPC `UNIMPLEMENTED`.

Note: Triton’s `ModelStreamInfer` is bidirectional streaming; Ray Serve’s gRPC proxy does not override stream-stream handlers, so this MVP does not support it.

## Proto generation

Required command (as requested):

### Proto generation inside the Ray container

If you’re using `rayproject/ray-llm:2.53.0-py311-cu128` (protobuf `4.25.x`), generate with a protobuf-4-compatible `grpcio-tools`:

```bash
podman-hpc run --rm --gpu --group-add keep-groups --userns=keep-id \
  -v "$PWD:/workspace:rw" -w /workspace \
  rayproject/ray-llm:2.53.0-py311-cu128 \
  /bin/bash -lc "pip -q install 'protobuf==4.25.8' 'grpcio-tools==1.59.0' && \
    python3 -m grpc_tools.protoc -I triton_compat/proto \
      --python_out triton_compat/generated \
      --grpc_python_out triton_compat/generated \
      triton_compat/proto/model_config.proto \
      triton_compat/proto/grpc_service.proto \
      triton_compat/proto/health.proto"
```

## Start the server

### Start inside the Ray container
This starts Ray + Serve and enables the gRPC proxy on port `8001`.

```bash
podman-hpc run --rm --gpu --group-add keep-groups --userns=keep-id \
  -p 8001:8001 -v "$PWD:/workspace:rw" -w /workspace \
  rayproject/ray-llm:2.53.0-py311-cu128 \
  /bin/bash -lc "python3 -m triton_compat.start"
```

## Build and run the Triton C++ gRPC client example

### Build `triton_client/src/c++` directly (requires system deps)

For nersc you can easily update cmake and install required deps via conda:
```bash
module load conda
conda install cmake grpcio grpcio-tools libgrpc
```

If you build `triton_client/src/c++` directly, you must have CMake packages for Protobuf, gRPC, and RapidJSON installed and discoverable (`Protobuf_DIR`, `gRPC_DIR`, `RapidJSON_DIR`, or `CMAKE_PREFIX_PATH`).

Configure and build:

```bash
cmake -S triton_client/src/c++ -B build/triton_client_cc \
  -DTRITON_ENABLE_CC_GRPC=ON \
  -DTRITON_ENABLE_EXAMPLES=ON \
  -DTRITON_REPO_ORGANIZATION=https://github.com/triton-inference-server \
  -DCMAKE_CXX_FLAGS="-Wno-unused-parameter -Wno-error=unused-parameter" \
  -DRAPIDJSON_INCLUDE_DIRS="$CONDA_PREFIX/include" 
cmake --build build/triton_client_cc -j 32
```

Run the examples against the Ray Serve gRPC proxy on `localhost:8001`:

```bash
./build/triton_client_cc/examples/simple_grpc_health_metadata -u localhost:8001
./build/triton_client_cc/examples/simple_grpc_infer_client -u localhost:8001
./build/triton_client_cc/examples/simple_grpc_async_infer_client -u localhost:8001
./build/triton_client_cc/examples/simple_grpc_custom_args_client -u localhost:8001
./build/triton_client_cc/examples/simple_grpc_keepalive_client -u localhost:8001
```