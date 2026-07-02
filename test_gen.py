#!/usr/bin/env python3
"""Standalone smoke test: одна генерация через RunPod -> out.png.
Проверяет, что ключ + эндпоинт живые, ДО запуска бота.

  RUNPOD_API_KEY=xxx python test_gen.py "your prompt here"
"""
import os
import sys
import time
import json
import base64
import httpx

KEY = os.environ.get("RUNPOD_API_KEY")
if not KEY:
    sys.exit("set RUNPOD_API_KEY (export RUNPOD_API_KEY=... or put in .env and `set -a; . ./.env`)")
EP = os.environ.get("RUNPOD_ENDPOINT_ID", "ti782qocbvn5f3")
BASE = f"https://api.runpod.ai/v2/{EP}"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "a beautiful young woman, photorealistic portrait, soft natural light"

wf = {
    "1":  {"inputs": {"unet_name": "krea2_turbo_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "13": {"inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2", "device": "default"}, "class_type": "CLIPLoader"},
    "4":  {"inputs": {"vae_name": "qwen_image_vae.safetensors"}, "class_type": "VAELoader"},
    "17": {"inputs": {"PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
                      "lora_1": {"on": True, "lora": "realism_engine_krea2_v2.safetensors", "strength": 1.0},
                      "model": ["1", 0], "clip": ["13", 0]}, "class_type": "Power Lora Loader (rgthree)"},
    "6":  {"inputs": {"text": PROMPT, "clip": ["17", 1]}, "class_type": "CLIPTextEncode"},
    "8":  {"inputs": {"conditioning": ["6", 0]}, "class_type": "ConditioningZeroOut"},
    "10": {"inputs": {"width": 832, "height": 1216, "batch_size": 1}, "class_type": "EmptyLatentImage"},
    "2":  {"inputs": {"seed": 12345, "steps": 8, "cfg": 1, "sampler_name": "er_sde", "scheduler": "sgm_uniform",
                      "denoise": 1, "model": ["17", 0], "positive": ["6", 0], "negative": ["8", 0],
                      "latent_image": ["10", 0]}, "class_type": "KSampler"},
    "3":  {"inputs": {"samples": ["2", 0], "vae": ["4", 0]}, "class_type": "VAEDecode"},
    "42": {"inputs": {"filename_prefix": "krea_test", "images": ["3", 0]}, "class_type": "SaveImage"},
}

h = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
print(f"endpoint={EP}  prompt={PROMPT!r}")
r = httpx.post(f"{BASE}/run", headers=h, json={"input": {"workflow": wf}}, timeout=60)
r.raise_for_status()
jid = r.json()["id"]
print("job:", jid)
for _ in range(180):
    time.sleep(2)
    js = httpx.get(f"{BASE}/status/{jid}", headers=h, timeout=30).json()
    print("status:", js.get("status"))
    st = js.get("status")
    if st == "COMPLETED":
        out = js["output"]
        imgs = out.get("images") if isinstance(out, dict) else out
        data = imgs[0]["data"].split(",")[-1]
        open("out.png", "wb").write(base64.b64decode(data))
        print("✅ saved out.png")
        break
    if st in ("FAILED", "CANCELLED", "TIMED_OUT"):
        print(json.dumps(js, indent=2)[:1500])
        break
