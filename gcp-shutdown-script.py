#!/home/ceshine/miniconda3/envs/pytorch/bin/python
import socket
import telegram

BOT_TOKEN = "559760930:AAGOgPA0OlqlFB7DrX0lyRc4Di3xeixdNO8"
CHAT_ID = "213781869"

bot = telegram.Bot(token=BOT_TOKEN)
host_name = socket.gethostname()
content = 'Machine name: %s is shutting down!' % host_name
bot.send_message(chat_id=CHAT_ID, text=content)
