"""Run this ONCE, locally, after you've created your Telegram bot and sent it
one message, to find your chat ID.

    python scripts/get_telegram_chat_id.py <YOUR_BOT_TOKEN>

It calls Telegram's public getUpdates endpoint and prints the chat ID(s) that
have messaged the bot. It does not store or transmit your token anywhere else.
"""
import sys
import requests

if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(1)

token = sys.argv[1]
resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10)
resp.raise_for_status()
data = resp.json()

if not data.get("ok"):
    print(f"Telegram API error: {data}")
    sys.exit(1)

results = data.get("result", [])
if not results:
    print("No messages found yet. Open your bot in Telegram and send it any "
          "message (e.g. 'hi'), then run this script again.")
    sys.exit(0)

seen = set()
for update in results:
    msg = update.get("message") or update.get("channel_post") or {}
    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    if chat_id is not None and chat_id not in seen:
        seen.add(chat_id)
        name = chat.get("username") or chat.get("first_name") or chat.get("title") or "?"
        print(f"chat_id={chat_id}  (from: {name}, type: {chat.get('type')})")

print("\nUse the chat_id above as your TELEGRAM_CHAT_ID secret.")
