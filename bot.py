#!/usr/bin/env python3
"""
Krea Prompt Lab — Telegram bot for marketers to test text->image prompts
against the RunPod serverless Krea 2 endpoint (txt2img, no face-swap).

Send a text prompt -> get an image. No limits. Params after "|".

Env:
  TG_BOT_TOKEN        Telegram bot token
  RUNPOD_API_KEY      RunPod API key
  RUNPOD_ENDPOINT_ID  Serverless endpoint id (default: ti782qocbvn5f3)
"""
import asyncio
import os
import base64
import random
import html
import json
import logging

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import (
    Message, CallbackQuery, BufferedInputFile,
    InlineKeyboardMarkup, InlineKeyboardButton,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kreabot")

TG_TOKEN = os.environ["TG_BOT_TOKEN"]
RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT = os.environ.get("RUNPOD_ENDPOINT_ID", "ti782qocbvn5f3")
BASE = f"https://api.runpod.ai/v2/{ENDPOINT}"

# ---- model / sampler config (matches the prod worker) ----
UNET = "krea2_turbo_fp8_scaled.safetensors"
CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
LORA = "realism_engine_krea2_v2.safetensors"
DEFAULTS = dict(w=832, h=1216, steps=8, lora=1.0, seed=None)


def clamp64(v, lo=512, hi=1536):
    v = max(lo, min(hi, int(v)))
    return (v // 64) * 64


def build_workflow(prompt, seed, w, h, steps, lora):
    """Krea 2 Turbo txt2img (no swap / no FaceDetailer) — fast, for prompt testing."""
    lora_node = {
        "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
        "lora_1": {"on": lora > 0, "lora": LORA, "strength": lora},
        "model": ["1", 0], "clip": ["13", 0],
    }
    return {
        "1":  {"inputs": {"unet_name": UNET, "weight_dtype": "default"}, "class_type": "UNETLoader"},
        "13": {"inputs": {"clip_name": CLIP, "type": "krea2", "device": "default"}, "class_type": "CLIPLoader"},
        "4":  {"inputs": {"vae_name": VAE}, "class_type": "VAELoader"},
        "17": {"inputs": lora_node, "class_type": "Power Lora Loader (rgthree)"},
        "6":  {"inputs": {"text": prompt, "clip": ["17", 1]}, "class_type": "CLIPTextEncode"},
        "8":  {"inputs": {"conditioning": ["6", 0]}, "class_type": "ConditioningZeroOut"},
        "10": {"inputs": {"width": w, "height": h, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "2":  {"inputs": {"seed": seed, "steps": steps, "cfg": 1, "sampler_name": "er_sde",
                          "scheduler": "sgm_uniform", "denoise": 1, "model": ["17", 0],
                          "positive": ["6", 0], "negative": ["8", 0], "latent_image": ["10", 0]},
               "class_type": "KSampler"},
        "3":  {"inputs": {"samples": ["2", 0], "vae": ["4", 0]}, "class_type": "VAEDecode"},
        "42": {"inputs": {"filename_prefix": "krea_test", "images": ["3", 0]}, "class_type": "SaveImage"},
    }


async def runpod_generate(workflow):
    headers = {"Authorization": f"Bearer {RUNPOD_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as c:
        r = await c.post(f"{BASE}/run", headers=headers, json={"input": {"workflow": workflow}})
        r.raise_for_status()
        job = r.json()
        jid = job.get("id")
        if not jid:
            raise RuntimeError(f"no job id in response: {job}")
        for _ in range(180):  # ~6 min max
            await asyncio.sleep(2)
            s = await c.get(f"{BASE}/status/{jid}", headers=headers)
            js = s.json()
            st = js.get("status")
            if st == "COMPLETED":
                return js.get("output")
            if st in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"{st}: {json.dumps(js)[:600]}")
        raise TimeoutError("generation timed out (>6 min)")


def extract_images(output):
    """Return list of (kind, value) where kind is 'b64' or 'url'."""
    items = []
    if isinstance(output, dict):
        items = output.get("images") or []
    elif isinstance(output, list):
        for o in output:
            if isinstance(o, dict) and o.get("images"):
                items += o["images"]
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        data = it.get("data")
        if it.get("type") in ("s3_url", "url"):
            out.append(("url", data or it.get("url")))
        elif data:
            out.append(("b64", data.split(",")[-1]))  # strip any data: prefix
    return out


def parse_message(text):
    params = dict(DEFAULTS)
    prompt = text
    if "|" in text:
        prompt, rest = text.split("|", 1)
        for tok in rest.split():
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            k = k.strip().lower()
            try:
                if k == "size" and "x" in v.lower():
                    a, b = v.lower().split("x")
                    params["w"], params["h"] = clamp64(a), clamp64(b)
                elif k in ("w", "width"):
                    params["w"] = clamp64(v)
                elif k in ("h", "height"):
                    params["h"] = clamp64(v)
                elif k == "seed":
                    params["seed"] = int(v)
                elif k == "steps":
                    params["steps"] = max(1, min(20, int(v)))
                elif k == "lora":
                    params["lora"] = max(0.0, min(3.0, float(v)))
            except ValueError:
                pass
    return prompt.strip(), params


LAST = {}  # user_id -> (prompt, params_with_resolved_seed)

bot = Bot(TG_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

HELP = (
    "🎨 <b>Krea Prompt Lab</b>\n\n"
    "Пришли <b>готовый промт на английском</b> — верну картинку с нашей "
    "нейронки (Krea 2 Turbo + realism LoRA). Без лимитов.\n\n"
    "💡 <b>Как писать под нашу модель:</b>\n"
    "• только <b>английский</b>\n"
    "• <b>связным описанием</b>, а не теги через запятую\n"
    "• опиши: кто · внешность · одежда · поза · сцена · свет · стиль\n\n"
    "<b>Примеры</b> (тапни по промту — скопируется, отправь его боту):\n"
    "<code>a young woman with red hair and freckles, wearing a green summer "
    "dress, sitting in a cozy cafe by the window, soft warm daylight, "
    "photorealistic, shallow depth of field</code>\n\n"
    "<code>a beautiful blonde woman in an elegant black evening dress, standing "
    "on a city rooftop at night, neon bokeh lights behind her, cinematic "
    "lighting, photorealistic, sharp focus</code>\n\n"
    "<code>a tanned brunette in a red bikini on a tropical beach at sunset, wet "
    "hair, golden hour light, ocean waves behind, photorealistic, natural skin "
    "texture</code>\n\n"
    "<b>Параметры</b> (опц., в конце после <code>|</code>):\n"
    "<code>...промт... | size=832x1216 seed=42 lora=1.2 steps=8</code>\n"
    "• <code>size=ШхВ</code> — разрешение (дефолт 832×1216, портрет)\n"
    "• <code>seed=N</code> — зафиксировать сид (повторяемость)\n"
    "• <code>lora=N</code> — сила realism-LoRA (дефолт 1.0, 0 = выкл)\n"
    "• <code>steps=N</code> — шаги (дефолт 8)\n\n"
    "Под картинкой — 🎲 новый сид · 🔁 тот же сид. В подписи виден seed — "
    "записывай удачные промпты."
)


def regen_kb():
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🎲 Новый сид", callback_data="re_new"),
        InlineKeyboardButton(text="🔁 Тот же сид", callback_data="re_same"),
    ]])


@dp.message(CommandStart())
async def cmd_start(m: Message):
    await m.answer(HELP)


@dp.message(Command("help"))
async def cmd_help(m: Message):
    await m.answer(HELP)


@dp.message(F.text)
async def on_text(m: Message):
    if m.text.startswith("/"):
        return  # ignore unknown commands
    prompt, params = parse_message(m.text)
    if not prompt:
        await m.answer("Пришли готовый промт на английском 🙂")
        return
    asyncio.create_task(do_gen(m.chat.id, m.from_user.id, prompt, params))


@dp.callback_query(F.data.in_({"re_new", "re_same"}))
async def on_regen(cb: CallbackQuery):
    saved = LAST.get(cb.from_user.id)
    if not saved:
        await cb.answer("Нет прошлого промта — пришли текст.", show_alert=True)
        return
    prompt, params = saved
    params = dict(params)
    if cb.data == "re_new":
        params["seed"] = None  # roll a fresh seed
    await cb.answer("Генерирую…")
    asyncio.create_task(do_gen(cb.message.chat.id, cb.from_user.id, prompt, params))


async def do_gen(chat_id, user_id, prompt, params):
    seed = params.get("seed")
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    params = {**params, "seed": seed}
    LAST[user_id] = (prompt, params)  # store resolved seed so "🔁 тот же сид" works
    w, h, steps, lora = params["w"], params["h"], params["steps"], params["lora"]

    status = await bot.send_message(chat_id, f"⏳ Генерирую… <code>seed {seed}</code>")
    try:
        wf = build_workflow(prompt, seed, w, h, steps, lora)
        output = await runpod_generate(wf)
        imgs = extract_images(output)
        if not imgs:
            raise RuntimeError(f"пустой ответ воркера: {json.dumps(output)[:400]}")
        cap = (f"🌱 <code>seed={seed}</code>  📐 {w}×{h}  ⚙️ steps {steps} · lora {lora}\n"
               f"📝 {html.escape(prompt[:850])}")
        for i, (kind, val) in enumerate(imgs):
            photo = BufferedInputFile(base64.b64decode(val), "gen.png") if kind == "b64" else val
            await bot.send_photo(
                chat_id, photo,
                caption=cap if i == 0 else None,
                reply_markup=regen_kb() if i == 0 else None,
            )
        await status.delete()
    except Exception as e:
        log.exception("gen failed")
        msg = f"❌ {html.escape(str(e))[:900]}"
        try:
            await status.edit_text(msg)
        except Exception:
            await bot.send_message(chat_id, msg)


async def main():
    log.info("Krea Prompt Lab starting | endpoint=%s", ENDPOINT)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
