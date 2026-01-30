from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import grpc
from ray import serve
from ray.serve.grpc_util import RayServegRPCContext

from triton_compat.backend import MODEL_NAME, MODEL_VERSION, infer_simple


def _ensure_generated_importable() -> None:
    # The generated modules import each other as top-level modules
    # (e.g. `import model_config_pb2`), so add the generated directory to sys.path.
    gen_dir = Path(__file__).resolve().parent / "generated"
    gen_path = str(gen_dir)
    if gen_path not in sys.path:
        sys.path.insert(0, gen_path)


_ensure_generated_importable()

import grpc_service_pb2  # noqa: E402
import health_pb2  # noqa: E402
import model_config_pb2  # noqa: E402


def _set_status_and_raise(
    grpc_context: Optional[RayServegRPCContext],
    code: grpc.StatusCode,
    details: str,
) -> None:
    # For Ray Serve gRPC proxy, it's enough to set the status code/details on the
    # context and return an empty response message. Raising an exception causes
    # noisy server-side stack traces for expected error cases (e.g. NOT_FOUND).
    if grpc_context is not None:
        grpc_context.set_code(code)
        grpc_context.set_details(details)


def _validate_model_name(
    grpc_context: Optional[RayServegRPCContext], model_name: str
) -> None:
    if model_name != MODEL_NAME:
        _set_status_and_raise(
            grpc_context,
            grpc.StatusCode.NOT_FOUND,
            f"Request for unknown model: '{model_name}' is not found",
        )
        raise _ModelNotFound()


class _ModelNotFound(Exception):
    pass


class _InvalidRequest(Exception):
    pass


class _Unimplemented(Exception):
    pass


def _stable_model_version(requested: str) -> str:
    return requested if requested else MODEL_VERSION


def _tensor_shape(spec_shape: Tuple[int, ...]) -> List[int]:
    return [int(d) for d in spec_shape]


def _build_model_config() -> model_config_pb2.ModelConfig:
    cfg = model_config_pb2.ModelConfig()
    cfg.name = MODEL_NAME
    cfg.platform = "ray_serve"
    cfg.max_batch_size = 0

    for name in ("INPUT0", "INPUT1"):
        t = cfg.input.add()
        t.name = name
        t.data_type = model_config_pb2.TYPE_INT32
        t.dims.extend([1, 16])

    for name in ("OUTPUT0", "OUTPUT1"):
        t = cfg.output.add()
        t.name = name
        t.data_type = model_config_pb2.TYPE_INT32
        t.dims.extend([1, 16])

    ig = cfg.instance_group.add()
    ig.kind = model_config_pb2.ModelInstanceGroup.KIND_CPU
    ig.count = 1
    return cfg


@serve.deployment
class TritonCompatDeployment:
    # -------------------------
    # Triton GRPCInferenceService (MVP)
    # -------------------------
    def ServerLive(
        self, request: grpc_service_pb2.ServerLiveRequest
    ) -> grpc_service_pb2.ServerLiveResponse:
        return grpc_service_pb2.ServerLiveResponse(live=True)

    def ServerReady(
        self, request: grpc_service_pb2.ServerReadyRequest
    ) -> grpc_service_pb2.ServerReadyResponse:
        return grpc_service_pb2.ServerReadyResponse(ready=True)

    def ModelReady(
        self,
        request: grpc_service_pb2.ModelReadyRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.ModelReadyResponse:
        try:
            _validate_model_name(grpc_context, request.name)
        except _ModelNotFound:
            return grpc_service_pb2.ModelReadyResponse()
        return grpc_service_pb2.ModelReadyResponse(ready=True)

    def ServerMetadata(
        self, request: grpc_service_pb2.ServerMetadataRequest
    ) -> grpc_service_pb2.ServerMetadataResponse:
        # Triton C++ example expects server_metadata.name == "triton".
        resp = grpc_service_pb2.ServerMetadataResponse()
        resp.name = "triton"
        resp.version = "0.0.0"
        return resp

    def ModelMetadata(
        self,
        request: grpc_service_pb2.ModelMetadataRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.ModelMetadataResponse:
        try:
            _validate_model_name(grpc_context, request.name)
        except _ModelNotFound:
            return grpc_service_pb2.ModelMetadataResponse()

        resp = grpc_service_pb2.ModelMetadataResponse()
        resp.name = MODEL_NAME
        resp.versions.append(MODEL_VERSION)
        resp.platform = "ray_serve"

        for name in ("INPUT0", "INPUT1"):
            t = resp.inputs.add()
            t.name = name
            t.datatype = "INT32"
            t.shape.extend([1, 16])

        for name in ("OUTPUT0", "OUTPUT1"):
            t = resp.outputs.add()
            t.name = name
            t.datatype = "INT32"
            t.shape.extend([1, 16])

        return resp

    def ModelConfig(
        self,
        request: grpc_service_pb2.ModelConfigRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.ModelConfigResponse:
        try:
            _validate_model_name(grpc_context, request.name)
        except _ModelNotFound:
            return grpc_service_pb2.ModelConfigResponse()

        resp = grpc_service_pb2.ModelConfigResponse()
        resp.config.CopyFrom(_build_model_config())
        return resp

    def ModelInfer(
        self,
        request: grpc_service_pb2.ModelInferRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.ModelInferResponse:
        try:
            _validate_model_name(grpc_context, request.model_name)
        except _ModelNotFound:
            return grpc_service_pb2.ModelInferResponse()

        if len(request.raw_input_contents) != len(request.inputs):
            _set_status_and_raise(
                grpc_context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "raw_input_contents count must match inputs count",
            )
            return grpc_service_pb2.ModelInferResponse()

        input_arrays: Dict[str, "object"] = {}
        for inp, raw in zip(request.inputs, request.raw_input_contents):
            if inp.parameters:
                # Shared memory and other advanced tensor params are out of scope.
                if any(k.startswith("shared_memory_") for k in inp.parameters.keys()):
                    _set_status_and_raise(
                        grpc_context,
                        grpc.StatusCode.UNIMPLEMENTED,
                        "Shared memory is not supported in MVP",
                    )
                    return grpc_service_pb2.ModelInferResponse()
            if inp.name not in ("INPUT0", "INPUT1"):
                _set_status_and_raise(
                    grpc_context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Unexpected input name: {inp.name}",
                )
                return grpc_service_pb2.ModelInferResponse()
            if inp.datatype != "INT32":
                _set_status_and_raise(
                    grpc_context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Unexpected datatype for {inp.name}: {inp.datatype}",
                )
                return grpc_service_pb2.ModelInferResponse()
            if list(inp.shape) != [1, 16]:
                _set_status_and_raise(
                    grpc_context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Unexpected shape for {inp.name}: {list(inp.shape)}",
                )
                return grpc_service_pb2.ModelInferResponse()

            # INT32 input (little-endian, row-major).
            if len(raw) != 1 * 16 * 4:
                _set_status_and_raise(
                    grpc_context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Raw buffer size mismatch for {inp.name}",
                )
                return grpc_service_pb2.ModelInferResponse()
            import numpy as np

            input_arrays[inp.name] = np.frombuffer(raw, dtype="<i4").reshape((1, 16))

        if "INPUT0" not in input_arrays or "INPUT1" not in input_arrays:
            _set_status_and_raise(
                grpc_context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "Both INPUT0 and INPUT1 are required",
            )
            return grpc_service_pb2.ModelInferResponse()

        outputs_np = infer_simple(input_arrays["INPUT0"], input_arrays["INPUT1"])

        requested_outputs = [o.name for o in request.outputs]
        if not requested_outputs:
            requested_outputs = ["OUTPUT0", "OUTPUT1"]

        resp = grpc_service_pb2.ModelInferResponse()
        resp.model_name = MODEL_NAME
        resp.model_version = _stable_model_version(request.model_version)
        resp.id = request.id
        resp.parameters["triton_final_response"].bool_param = True

        for out_name in requested_outputs:
            if out_name not in ("OUTPUT0", "OUTPUT1"):
                _set_status_and_raise(
                    grpc_context,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Unexpected requested output name: {out_name}",
                )
                return grpc_service_pb2.ModelInferResponse()
            out_arr = outputs_np[out_name]
            ot = resp.outputs.add()
            ot.name = out_name
            ot.datatype = "INT32"
            ot.shape.extend([1, 16])
            import numpy as np

            resp.raw_output_contents.append(
                np.asarray(out_arr, dtype="<i4", order="C").tobytes(order="C")
            )

        return resp

    # -------------------------
    # gRPC health.v1 Health service (optional)
    # -------------------------
    def Check(
        self,
        request: health_pb2.HealthCheckRequest,
    ) -> health_pb2.HealthCheckResponse:
        # Conservative: always SERVING while the process is up.
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING
        )

    # -------------------------
    # Non-MVP Triton RPCs (must be UNIMPLEMENTED)
    # -------------------------
    def _unimplemented(self, grpc_context: Optional[RayServegRPCContext], method: str):
        _set_status_and_raise(
            grpc_context,
            grpc.StatusCode.UNIMPLEMENTED,
            f"{method} is not implemented in MVP",
        )
        raise _Unimplemented()

    def ModelStatistics(
        self,
        request: grpc_service_pb2.ModelStatisticsRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.ModelStatisticsResponse:
        try:
            self._unimplemented(grpc_context, "ModelStatistics")
        except _Unimplemented:
            return grpc_service_pb2.ModelStatisticsResponse()

    def RepositoryIndex(
        self,
        request: grpc_service_pb2.RepositoryIndexRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.RepositoryIndexResponse:
        try:
            self._unimplemented(grpc_context, "RepositoryIndex")
        except _Unimplemented:
            return grpc_service_pb2.RepositoryIndexResponse()

    def RepositoryModelLoad(
        self,
        request: grpc_service_pb2.RepositoryModelLoadRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.RepositoryModelLoadResponse:
        try:
            self._unimplemented(grpc_context, "RepositoryModelLoad")
        except _Unimplemented:
            return grpc_service_pb2.RepositoryModelLoadResponse()

    def RepositoryModelUnload(
        self,
        request: grpc_service_pb2.RepositoryModelUnloadRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.RepositoryModelUnloadResponse:
        try:
            self._unimplemented(grpc_context, "RepositoryModelUnload")
        except _Unimplemented:
            return grpc_service_pb2.RepositoryModelUnloadResponse()

    def SystemSharedMemoryStatus(
        self,
        request: grpc_service_pb2.SystemSharedMemoryStatusRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.SystemSharedMemoryStatusResponse:
        try:
            self._unimplemented(grpc_context, "SystemSharedMemoryStatus")
        except _Unimplemented:
            return grpc_service_pb2.SystemSharedMemoryStatusResponse()

    def SystemSharedMemoryRegister(
        self,
        request: grpc_service_pb2.SystemSharedMemoryRegisterRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.SystemSharedMemoryRegisterResponse:
        try:
            self._unimplemented(grpc_context, "SystemSharedMemoryRegister")
        except _Unimplemented:
            return grpc_service_pb2.SystemSharedMemoryRegisterResponse()

    def SystemSharedMemoryUnregister(
        self,
        request: grpc_service_pb2.SystemSharedMemoryUnregisterRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.SystemSharedMemoryUnregisterResponse:
        try:
            self._unimplemented(grpc_context, "SystemSharedMemoryUnregister")
        except _Unimplemented:
            return grpc_service_pb2.SystemSharedMemoryUnregisterResponse()

    def CudaSharedMemoryStatus(
        self,
        request: grpc_service_pb2.CudaSharedMemoryStatusRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.CudaSharedMemoryStatusResponse:
        try:
            self._unimplemented(grpc_context, "CudaSharedMemoryStatus")
        except _Unimplemented:
            return grpc_service_pb2.CudaSharedMemoryStatusResponse()

    def CudaSharedMemoryRegister(
        self,
        request: grpc_service_pb2.CudaSharedMemoryRegisterRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.CudaSharedMemoryRegisterResponse:
        try:
            self._unimplemented(grpc_context, "CudaSharedMemoryRegister")
        except _Unimplemented:
            return grpc_service_pb2.CudaSharedMemoryRegisterResponse()

    def CudaSharedMemoryUnregister(
        self,
        request: grpc_service_pb2.CudaSharedMemoryUnregisterRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.CudaSharedMemoryUnregisterResponse:
        try:
            self._unimplemented(grpc_context, "CudaSharedMemoryUnregister")
        except _Unimplemented:
            return grpc_service_pb2.CudaSharedMemoryUnregisterResponse()

    def TraceSetting(
        self,
        request: grpc_service_pb2.TraceSettingRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.TraceSettingResponse:
        try:
            self._unimplemented(grpc_context, "TraceSetting")
        except _Unimplemented:
            return grpc_service_pb2.TraceSettingResponse()

    def LogSettings(
        self,
        request: grpc_service_pb2.LogSettingsRequest,
        grpc_context: Optional[RayServegRPCContext] = None,
    ) -> grpc_service_pb2.LogSettingsResponse:
        try:
            self._unimplemented(grpc_context, "LogSettings")
        except _Unimplemented:
            return grpc_service_pb2.LogSettingsResponse()

    # NOTE: Triton defines ModelStreamInfer as bidirectional streaming.
    # Ray Serve's gRPC proxy does not override stream-stream handlers, so this
    # method may not be invoked. It is included for completeness.
    def ModelStreamInfer(self, *args, **kwargs):
        grpc_context = kwargs.get("grpc_context")
        try:
            self._unimplemented(grpc_context, "ModelStreamInfer")
        except _Unimplemented:
            # Triton ModelStreamInfer returns ModelStreamInferResponse on the wire,
            # but this method isn't expected to be invoked in this MVP.
            return None


app = TritonCompatDeployment.bind()
