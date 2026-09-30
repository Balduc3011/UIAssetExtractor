"""ONNX Runtime providers + one-model-at-a-time manager (keeps VRAM low)."""
import gc
import threading

_preloaded = False
_lock = threading.Lock()


def _ort():
    global _preloaded
    import onnxruntime as ort
    if not _preloaded:
        _preloaded = True
        # Load CUDA/cuDNN DLLs shipped as pip packages (onnxruntime-gpu[cuda,cudnn]).
        try:
            ort.preload_dlls()
        except Exception:
            pass
    return ort


def status() -> dict:
    try:
        ort = _ort()
        prov = ort.get_available_providers()
        return {
            "onnxruntime": ort.__version__,
            "cuda_available": "CUDAExecutionProvider" in prov,
            "providers": prov,
        }
    except Exception as e:
        return {"error": str(e), "cuda_available": False, "providers": []}


def providers(mode: str) -> list:
    if mode == "off":
        return ["CPUExecutionProvider"]
    ort = _ort()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return ["CPUExecutionProvider"]
    opts = {
        "device_id": 0,
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "HEURISTIC",
    }
    if mode == "eco":
        opts["gpu_mem_limit"] = 2 * 1024 ** 3
    return [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]


class ModelManager:
    """Holds at most ONE heavy model; released after each job."""

    def __init__(self):
        self._key = None
        self._obj = None

    def rembg(self, model: str, mode: str):
        key = ("rembg", model, mode)
        with _lock:
            if self._key != key:
                self.release()
                from rembg import new_session
                self._obj = new_session(model, providers=providers(mode))
                self._key = key
            return self._obj

    def active_provider(self) -> str | None:
        try:
            return self._obj.inner_session.get_providers()[0]
        except Exception:
            return None

    def release(self):
        self._obj = None
        self._key = None
        gc.collect()


models = ModelManager()

_ocr = None


def ocr():
    """RapidOCR on CPU (small, fast; avoids GPU/CUDA conflicts)."""
    global _ocr
    if _ocr is None:
        from rapidocr import RapidOCR
        _ocr = RapidOCR()
    return _ocr
