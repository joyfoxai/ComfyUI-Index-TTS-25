import gc
import importlib
import os
import random
import re
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import folder_paths
import numpy as np
import torch
import torchaudio

try:
    import comfy.model_management as model_management
except Exception:
    model_management = None


HERE = Path(__file__).resolve().parent
COMFY_ROOT = HERE.parents[1]
DEFAULT_MODELS_ROOT = Path(folder_paths.models_dir) / "indextts"
SHARED_MODELS_ROOT = Path("/root/comfyui_models/indextts")
MODEL_TYPE = "indextts"
MODEL_LOCK = threading.RLock()
MODEL_CACHE = {"key": None, "model": None}

LANGUAGES = ["ZH", "EN", "JA", "AR", "ES"]
EMOTION_MODES = [
    "same_as_speaker",
    "reference_audio",
    "emotion_vector",
    "emotion_text",
    "extra_emotion_text",
]


@dataclass(frozen=True)
class IndexTTS25ModelHandle:
    model_name: str
    precision: str
    device: str
    cuda_kernel: bool
    deepspeed: bool
    gpt_accel: bool
    torch_compile: bool

    def get_model(self):
        return _get_model(
            self.model_name,
            self.precision,
            self.device,
            self.cuda_kernel,
            self.deepspeed,
            self.gpt_accel,
            self.torch_compile,
        )


def _register_model_folder():
    DEFAULT_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    for index, root in enumerate(_model_roots()):
        if index > 0 and not root.parent.is_dir():
            continue
        try:
            folder_paths.add_model_folder_path(
                MODEL_TYPE, str(root), is_default=(root == DEFAULT_MODELS_ROOT)
            )
        except TypeError:
            folder_paths.add_model_folder_path(MODEL_TYPE, str(root))


def _model_roots():
    roots = []
    configured = os.environ.get("INDEXTTS_MODELS_DIR")
    if configured:
        configured_path = Path(configured).expanduser()
        roots.append(
            configured_path if configured_path.name == "indextts" else configured_path / "indextts"
        )
    roots.extend([SHARED_MODELS_ROOT, DEFAULT_MODELS_ROOT])
    unique = []
    for root in roots:
        root = root.resolve() if root.exists() else root.absolute()
        if root not in unique:
            unique.append(root)
    return unique


_register_model_folder()


def _model_choices():
    choices = []
    for root in _model_roots():
        if (root / "config.yaml").is_file():
            choices.append(".")
        if root.is_dir():
            for config_path in root.glob("*/config.yaml"):
                choices.append(config_path.parent.name)
    choices.extend(["IndexTTS-2", "IndexTTS-2.5"])
    return sorted(set(choices))


def _resolve_model_dir(model_name):
    checked = []
    for root in _model_roots():
        model_dir = root if model_name == "." else root / model_name
        checked.append(str(model_dir))
        if not all((model_dir / name).is_file() for name in ("config.yaml", "gpt.pth", "s2mel.pth")):
            continue
        version = _detect_model_version(model_dir)
        required = (
            ("codec.pth", "wav2vec2bert_stats.pt")
            if version == "2.5"
            else ("bpe.model", "wav2vec2bert_stats.pt")
        )
        if all((model_dir / name).is_file() for name in required):
            return model_dir.resolve()
    raise FileNotFoundError(
        "IndexTTS 2/2.5 model was not found or is incomplete. Checked: "
        + ", ".join(checked)
    )


def _detect_model_version(model_dir):
    config_path = Path(model_dir) / "config.yaml"
    match = re.search(
        r"^\s*version\s*:\s*['\"]?([0-9.]+)",
        config_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Model config has no version field: {config_path}")
    if match.group(1).startswith("2.5"):
        return "2.5"
    if match.group(1).startswith("2"):
        return "2"
    raise ValueError(f"Unsupported IndexTTS model version {match.group(1)} in {config_path}")


def _resolve_indextts_repo():
    candidates = []
    configured = os.environ.get("INDEXTTS_REPO")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([
        HERE / "index-tts",
        COMFY_ROOT.parent / "index-tts",
    ])
    for candidate in candidates:
        if (candidate / "indextts" / "infer_v2_5.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "IndexTTS source repository was not found. Set INDEXTTS_REPO or place "
        "the repository next to ComfyUI as '../index-tts'."
    )


def _import_indextts(model_version="2.5"):
    repo = _resolve_indextts_repo()
    repo_str = str(repo)
    # ComfyUI can load another IndexTTS custom node first. Since both projects
    # use the top-level package name ``indextts``, an older package already in
    # sys.modules would hide this node's bundled 2.5 source even after changing
    # sys.path. Put our source first and discard only the conflicting package
    # modules before importing IndexTTS 2.5.
    sys.path[:] = [entry for entry in sys.path if entry != repo_str]
    sys.path.insert(0, repo_str)
    bundled_package = (repo / "indextts").resolve()
    loaded_package = sys.modules.get("indextts")
    if loaded_package is not None:
        loaded_file = getattr(loaded_package, "__file__", None)
        loaded_paths = [Path(path).resolve() for path in getattr(loaded_package, "__path__", [])]
        is_bundled = bundled_package in loaded_paths
        if loaded_file:
            try:
                Path(loaded_file).resolve().relative_to(bundled_package)
                is_bundled = True
            except ValueError:
                pass
        if not is_bundled:
            for module_name in list(sys.modules):
                if module_name == "indextts" or module_name.startswith("indextts."):
                    sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    try:
        if model_version == "2.5":
            from indextts.infer_v2_5 import IndexTTS2
        else:
            from indextts.infer_v2 import IndexTTS2
    except ModuleNotFoundError as exc:
        if exc.name == "indextts.infer_v2_5":
            raise RuntimeError(
                f"The bundled IndexTTS 2.5 source could not be imported from {repo}. "
                "Check that index-tts/indextts/infer_v2_5.py exists, then restart ComfyUI."
            ) from exc
        raise RuntimeError(
            f"IndexTTS dependency '{exc.name}' is missing from the ComfyUI Python "
            "environment. Install the dependencies listed in this node's "
            "requirements.txt, then restart ComfyUI."
        ) from exc
    return IndexTTS2


def _clear_model():
    model = MODEL_CACHE.get("model")
    MODEL_CACHE["model"] = None
    MODEL_CACHE["key"] = None
    if model is not None:
        try:
            model.gpt.to("cpu")
        except Exception:
            pass
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if model_management is not None:
        try:
            model_management.soft_empty_cache()
        except Exception:
            pass


def _get_model(
    model_name,
    precision,
    device,
    cuda_kernel,
    deepspeed,
    gpt_accel,
    torch_compile,
):
    model_dir = _resolve_model_dir(model_name)
    model_version = _detect_model_version(model_dir)
    resolved_device = None if device == "auto" else device
    key = (
        str(model_dir), model_version, precision, device, bool(cuda_kernel), bool(deepspeed),
        bool(gpt_accel), bool(torch_compile),
    )
    if MODEL_CACHE["model"] is not None and MODEL_CACHE["key"] == key:
        return MODEL_CACHE["model"]

    _clear_model()
    IndexTTS2 = _import_indextts(model_version)
    common_kwargs = dict(
        cfg_path=str(model_dir / "config.yaml"), model_dir=str(model_dir),
        device=resolved_device, use_cuda_kernel=bool(cuda_kernel),
        use_deepspeed=bool(deepspeed), use_accel=bool(gpt_accel),
        use_torch_compile=bool(torch_compile), use_qwen_emo=True,
    )
    if model_version == "2.5":
        model = IndexTTS2(
            use_bf16=(precision == "bf16"), use_gpt_latent=False, **common_kwargs
        )
    else:
        model = IndexTTS2(use_fp16=(precision == "bf16"), **common_kwargs)
    model._comfy_indextts_version = model_version
    MODEL_CACHE["key"] = key
    MODEL_CACHE["model"] = model
    return model


def _save_comfy_audio(audio, prefix):
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError(f"{prefix} must be a ComfyUI AUDIO value")
    waveform = audio["waveform"].detach().cpu().float()
    if waveform.ndim == 3:
        waveform = waveform[0]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"{prefix} waveform must have shape [B,C,T], [C,T], or [T]")
    waveform = torch.nan_to_num(waveform).clamp(-1.0, 1.0)
    handle = tempfile.NamedTemporaryFile(prefix=f"indextts_{prefix}_", suffix=".wav", delete=False)
    path = handle.name
    handle.close()
    torchaudio.save(path, waveform, int(audio["sample_rate"]))
    return path


def _load_output_audio(path):
    waveform, sample_rate = torchaudio.load(path)
    return {"waveform": waveform.float().unsqueeze(0), "sample_rate": int(sample_rate)}


def _comfy_audio_from_infer_result(result):
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"Unexpected IndexTTS inference result: {type(result).__name__}")
    sample_rate, samples = result
    waveform = torch.as_tensor(samples)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        # IndexTTS returns NumPy audio as [T, C]; ComfyUI expects [B, C, T].
        waveform = waveform.transpose(0, 1)
    else:
        raise RuntimeError(f"Unexpected IndexTTS audio shape: {tuple(waveform.shape)}")
    if not waveform.is_floating_point():
        waveform = waveform.float() / 32768.0
    else:
        waveform = waveform.float()
    return {
        "waveform": waveform.clamp(-1.0, 1.0).unsqueeze(0),
        "sample_rate": int(sample_rate),
    }


class IndexTTS25ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_model_choices(), {"default": "IndexTTS-2.5"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "device": (["auto", "cuda:0", "cpu"], {"default": "auto"}),
                "cuda_kernel": ("BOOLEAN", {"default": False}),
                "deepspeed": ("BOOLEAN", {"default": False}),
                "gpt_accel": ("BOOLEAN", {"default": False}),
                "torch_compile": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INDEXTTS25_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/IndexTTS"
    DESCRIPTION = "Load and cache an IndexTTS 2 or 2.5 model for synthesis nodes."

    def load_model(
        self, model_name, precision, device, cuda_kernel, deepspeed,
        gpt_accel, torch_compile,
    ):
        handle = IndexTTS25ModelHandle(
            model_name=model_name,
            precision=precision,
            device=device,
            cuda_kernel=bool(cuda_kernel),
            deepspeed=bool(deepspeed),
            gpt_accel=bool(gpt_accel),
            torch_compile=bool(torch_compile),
        )
        with MODEL_LOCK:
            handle.get_model()
        return (handle,)


class IndexTTS25Synthesize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("INDEXTTS25_MODEL",),
                "speaker_audio": ("AUDIO",),
                "text": ("STRING", {"multiline": True, "default": "欢迎使用 IndexTTS 2.5。"}),
                "language": (LANGUAGES, {"default": "ZH"}),
                "emotion_mode": (EMOTION_MODES, {"default": "same_as_speaker"}),
                "emotion_weight": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "extra_emotion_text": ("STRING", {"multiline": True, "default": ""}),
                "happy": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "angry": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sad": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "fearful": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "disgusted": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "melancholic": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "surprised": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "calm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "emotion_random": ("BOOLEAN", {"default": False}),
                "duration_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "interval_silence_ms": ("INT", {"default": 200, "min": 0, "max": 5000, "step": 10}),
                "text_normalization": ("BOOLEAN", {"default": True}),
                "max_text_tokens_per_segment": ("INT", {"default": 120, "min": 20, "max": 400, "step": 2}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "length_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "max_mel_tokens": ("INT", {"default": 1500, "min": 50, "max": 4096, "step": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
            },
            "optional": {
                "emotion_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "synthesize"
    CATEGORY = "audio/IndexTTS"
    DESCRIPTION = "IndexTTS 2/2.5 zero-shot multilingual and emotion-controllable speech synthesis."

    def synthesize(
        self, model, speaker_audio, text, language, emotion_mode, emotion_weight,
        extra_emotion_text, happy, angry, sad, fearful, disgusted, melancholic, surprised,
        calm, emotion_random, duration_factor, interval_silence_ms, text_normalization,
        max_text_tokens_per_segment, do_sample, temperature, top_p, top_k, num_beams,
        repetition_penalty, length_penalty, max_mel_tokens, seed, emotion_audio=None,
    ):
        if not isinstance(model, IndexTTS25ModelHandle):
            raise TypeError("model must come from an IndexTTS 2/2.5 Model Loader node")
        if not text or not text.strip():
            raise ValueError("Text must not be empty")
        if emotion_mode == "reference_audio" and emotion_audio is None:
            raise ValueError("emotion_audio must be connected in reference_audio mode")
        if emotion_mode == "extra_emotion_text" and not extra_emotion_text.strip():
            raise ValueError("extra_emotion_text must not be empty in extra_emotion_text mode")

        speaker_path = _save_comfy_audio(speaker_audio, "speaker")
        emotion_path = None

        try:
            if emotion_mode == "reference_audio":
                emotion_path = _save_comfy_audio(emotion_audio, "emotion")

            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            with MODEL_LOCK:
                tts = model.get_model()
                model_version = getattr(tts, "_comfy_indextts_version", "2.5")
                if model_version == "2" and language not in {"ZH", "EN"}:
                    raise ValueError(
                        f"IndexTTS 2 supports ZH/EN in this node; {language} requires IndexTTS 2.5"
                    )
                emotion_vector = None
                if emotion_mode == "emotion_vector":
                    emotion_vector = tts.normalize_emo_vec(
                        [happy, angry, sad, fearful, disgusted, melancholic, surprised, calm],
                        apply_bias=True,
                    )

                infer_kwargs = dict(
                    spk_audio_prompt=speaker_path,
                    text=text.strip(),
                    # TorchAudio 2.9 changed integer PCM save semantics and can
                    # turn IndexTTS's int16 tensor into a fully clipped WAV.
                    # Consume the native (sample_rate, int16 ndarray) result
                    # and normalize it explicitly for ComfyUI instead.
                    output_path=None,
                    emo_audio_prompt=emotion_path,
                    emo_alpha=float(emotion_weight),
                    emo_vector=emotion_vector,
                    use_emo_text=(emotion_mode in {"emotion_text", "extra_emotion_text"}),
                    emo_text=(
                        extra_emotion_text.strip()
                        if emotion_mode == "extra_emotion_text"
                        else None
                    ),
                    use_random=bool(emotion_random),
                    interval_silence=int(interval_silence_ms),
                    max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                    do_sample=bool(do_sample),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=(int(top_k) if int(top_k) > 0 else None),
                    num_beams=int(num_beams),
                    repetition_penalty=float(repetition_penalty),
                    length_penalty=float(length_penalty),
                    max_mel_tokens=int(max_mel_tokens),
                )
                if model_version == "2.5":
                    infer_kwargs.update(
                        lang=language,
                        duration_factor=float(duration_factor),
                        text_normalization=bool(text_normalization),
                    )
                result = tts.infer(**infer_kwargs)
            return (_comfy_audio_from_infer_result(result),)
        finally:
            for path in (speaker_path, emotion_path):
                if path:
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass


class IndexTTS25Unload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload_model"
    CATEGORY = "audio/IndexTTS"
    DESCRIPTION = "Unload the cached IndexTTS 2/2.5 model and release GPU memory."

    def unload_model(self, unload):
        if unload:
            with MODEL_LOCK:
                _clear_model()
            return ("IndexTTS 2/2.5 model unloaded",)
        return ("IndexTTS 2/2.5 model kept loaded",)


NODE_CLASS_MAPPINGS = {
    "IndexTTS25ModelLoader": IndexTTS25ModelLoader,
    "IndexTTS25Synthesize": IndexTTS25Synthesize,
    "IndexTTS25Unload": IndexTTS25Unload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS25ModelLoader": "IndexTTS 2 / 2.5 Model Loader",
    "IndexTTS25Synthesize": "IndexTTS 2 / 2.5 Synthesize",
    "IndexTTS25Unload": "IndexTTS 2 / 2.5 Unload Model",
}
