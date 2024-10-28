from fastapi import FastAPI
from .bot_manager import app as bot_manager_app

app = FastAPI()

# Mount bot manager under /bot
app.mount("/bot", bot_manager_app)

@app.get("/")
async def root():
    return {"message": "Welcome to the trading bot API"}
