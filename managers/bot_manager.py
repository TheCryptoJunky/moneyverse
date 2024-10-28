from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

# Pydantic model for bot control requests
class BotAction(BaseModel):
    bot_id: str

# Initialize FastAPI app
app = FastAPI()

# Sample bot actions (replace with actual logic)
bots = {
    "bot1": {"status": "stopped"},
    "bot2": {"status": "stopped"},
}

@app.post("/start_bot")
async def start_bot(action: BotAction):
    """Starts the specified bot."""
    bot_id = action.bot_id
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    # Async logic for starting the bot
    bots[bot_id]["status"] = "running"
    # Replace this with actual bot starting logic (async tasks, etc.)
    await asyncio.sleep(1)  # Simulate delay
    return {"status": f"{bot_id} started"}

@app.post("/stop_bot")
async def stop_bot(action: BotAction):
    """Stops the specified bot."""
    bot_id = action.bot_id
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    # Async logic for stopping the bot
    bots[bot_id]["status"] = "stopped"
    await asyncio.sleep(1)  # Simulate delay
    return {"status": f"{bot_id} stopped"}

@app.get("/status")
async def get_status(bot_id: str):
    """Get the current status of a bot."""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    return {"bot_id": bot_id, "status": bots[bot_id]["status"]}
