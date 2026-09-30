"""Run by setup.bat: import everything and pre-download the OCR models."""
import sys

import numpy as np


def main():
    from . import gpu
    st = gpu.status()
    print("  ONNX Runtime:", st.get("onnxruntime"), "| CUDA:", st.get("cuda_available"))
    import cv2, fastapi, rembg, rectpack, imagehash, anthropic, openai  # noqa: F401
    print("  Thu vien: OK")
    try:
        img = np.full((64, 200, 3), 255, np.uint8)
        cv2.putText(img, "TEST", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
        gpu.ocr()(img)
        print("  OCR model: OK")
    except Exception as e:
        print("  OCR model: loi (se tai lai khi chay):", e)
    if "--rembg" in sys.argv:
        from .config import load_settings
        s = load_settings()
        gpu.models.rembg(s["ai_matte_model"], s["gpu_mode"])
        print("  rembg model:", s["ai_matte_model"], "OK |", gpu.models.active_provider())


if __name__ == "__main__":
    main()
