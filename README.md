# Krea Prompt Lab — Telegram bot

Маркетолог шлёт промт текстом → бот возвращает картинку с serverless-эндпоинта RunPod
(**Krea 2 Turbo + realism LoRA, txt2img, без свапа**). Без лимитов — для подбора промптов
под нашу нейронку.

Отдельный лёгкий процесс: только polling Telegram + прокси в RunPod. Прод-инфру не трогает.

## Деплой на сервере

```bash
git clone https://github.com/soldatsc/storytestb.git
cd storytestb
cp .env.example .env
nano .env          # вписать TG_BOT_TOKEN и RUNPOD_API_KEY
```

### Вариант A — Docker (рекомендую, если на сервере есть docker)
```bash
docker build -t krea-prompt-bot .
docker run -d --restart unless-stopped --env-file .env --name krea-prompt-bot krea-prompt-bot
docker logs -f krea-prompt-bot        # проверить, что стартанул
```

### Вариант B — pm2 / systemd (если docker нет)
```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
set -a; . ./.env; set +a              # подгрузить .env в окружение
pm2 start "python bot.py" --name krea-prompt-bot
pm2 save
```

## Смоук-тест (проверить эндпоинт ДО бота)
```bash
set -a; . ./.env; set +a
python test_gen.py "beautiful woman, photorealistic portrait"   # -> out.png
```

## Использование
- `/start` — помощь
- Пришли промт текстом → картинка
- Параметры после `|`:
  `девушка в красном платье | size=832x1216 seed=42 lora=1.2 steps=8`
- Кнопки под картинкой: 🎲 новый сид, 🔁 тот же сид
- В подписи — seed + параметры (записывать, что сработало)

## Дефолты (правятся в `bot.py`, `DEFAULTS`)
- `832×1216` (портрет). Прод-точность → `size=1080x1920` (медленнее).
- 8 шагов, CFG 1, er_sde / sgm_uniform (Krea Turbo). realism LoRA @ 1.0.

## Безопасность
- Секреты только в `.env` (в `.gitignore`, в публичный репо не попадают).
- Скорость упирается в GPU эндпоинта (MIG-слайс медленнее полного 4090).
