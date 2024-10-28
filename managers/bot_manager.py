# Full file path: /moneyverse/managers/bot_manager.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from centralized_logger import CentralizedLogger

# Pydantic model for bot control requests
class BotAction(BaseModel):
    bot_id: str

# Initialize FastAPI app and centralized logger
app = FastAPI()
logger = CentralizedLogger()

# Sample bot status dictionary (replace with actual bot instances and logic)
bots = {
    "bot1": {"status": "stopped", "strategy": "Arbitrage", "performance": {}},
    "bot2": {"status": "stopped", "strategy": "Sniper", "performance": {}},
}

@app.post("/start_bot")
async def start_bot(action: BotAction):
    """Starts the specified bot."""
    bot_id = action.bot_id
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")

    if bots[bot_id]["status"] == "running":
        raise HTTPException(status_code=400, detail="Bot is already running")

    # Simulate async bot start with real bot logic
    bots[bot_id]["status"] = "running"
    logger.log("info", f"Starting bot {bot_id} with strategy {bots[bot_id]['strategy']}")
    await asyncio.sleep(1)  # Placeholder for actual async start logic
    return {"status": f"{bot_id} started"}

@app.post("/stop_bot")
async def stop_bot(action: BotAction):
    """Stops the specified bot."""
    bot_id = action.bot_id
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")

    if bots[bot_id]["status"] == "stopped":
        raise HTTPException(status_code=400, detail="Bot is already stopped")

    # Simulate async bot stop with real bot logic
    bots[bot_id]["status"] = "stopped"
    logger.log("info", f"Stopping bot {bot_id}")
    await asyncio.sleep(1)  # Placeholder for actual async stop logic
    return {"status": f"{bot_id} stopped"}

@app.get("/status")
async def get_status(bot_id: str):
    """Get the current status of a bot."""
    if bot_id not in bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    bot_status = bots[bot_id]
    logger.log("info", f"Checked status for bot {bot_id}: {bot_status['status']}")
    return {"bot_id": bot_id, "status": bot_status["status"], "strategy": bot_status["strategy"], "performance": bot_status["performance"]}

@app.get("/all_statuses")
async def get_all_statuses():
    """Get the status of all bots."""
    logger.log("info", "Retrieved status of all bots")
    return [{"bot_id": bot_id, "status": info["status"], "strategy": info["strategy"]} for bot_id, info in bots.items()]
