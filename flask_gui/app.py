# Full file path: /moneyverse/flask_gui/app.py

from flask import Flask, jsonify, request
from moneyverse.managers.wallet_manager import WalletManager
from datetime import datetime

app = Flask(__name__)
wallet_manager = WalletManager()

# Transaction statistics storage
transaction_stats = {
    "most_expensive": {"time": None, "aggregator": None, "cost": 0},
    "least_expensive": {"time": None, "aggregator": None, "cost": float('inf')}
}

# --- Wallet Management and Rebalancing Endpoints ---

@app.route("/wallets", methods=["GET"])
def get_wallets():
    """Retrieve all wallet details including balance and status."""
    return jsonify(wallet_manager.wallets)

@app.route("/wallets/add", methods=["POST"])
def add_wallet():
    """Dynamically add a new wallet with initial balance."""
    data = request.json
    wallet_id = data["wallet_id"]
    balance = data["balance"]
    wallet_manager.wallets.append({
        "wallet_id": wallet_id,
        "balance": balance,
        "status": "active",
        "tokens": []
    })
    return jsonify({"status": f"Wallet {wallet_id} added with balance {balance}."})

@app.route("/wallets/remove", methods=["POST"])
def remove_wallet():
    """Remove a wallet by its ID."""
    data = request.json
    wallet_id = data["wallet_id"]
    wallet_manager.wallets = [w for w in wallet_manager.wallets if w["wallet_id"] != wallet_id]
    return jsonify({"status": f"Wallet {wallet_id} removed."})

@app.route("/rebalance", methods=["POST"])
async def rebalance_wallets():
    """
    Trigger AI-driven rebalancing for all wallets.
    Automatically updates most/least expensive transactions.
    """
    await wallet_manager.ai_autonomous_rebalance()
    return jsonify({"status": "Rebalancing executed"})

@app.route("/transaction-stats", methods=["GET"])
def get_transaction_stats():
    """Retrieve statistics for the most and least expensive transactions."""
    return jsonify(transaction_stats)

@app.route("/transaction-stats/update", methods=["POST"])
def update_transaction_stats():
    """
    Update transaction stats after a rebalancing operation.
    Automatically compares cost and updates most/least expensive records.
    """
    data = request.json
    transaction_cost = data["cost"]
    aggregator = data["aggregator"]
    time_of_transaction = datetime.now().isoformat()

    # Update most expensive transaction
    if transaction_cost > transaction_stats["most_expensive"]["cost"]:
        transaction_stats["most_expensive"] = {
            "time": time_of_transaction,
            "aggregator": aggregator,
            "cost": transaction_cost
        }

    # Update least expensive transaction
    if transaction_cost < transaction_stats["least_expensive"]["cost"]:
        transaction_stats["least_expensive"] = {
            "time": time_of_transaction,
            "aggregator": aggregator,
            "cost": transaction_cost
        }

    return jsonify({"status": "Transaction stats updated."})

# --- Additional Endpoints for Asset Management ---

@app.route("/wallets/stable-holdings", methods=["GET"])
def stable_holdings():
    """View wallets holding stable coins for lower volatility storage."""
    stable_wallets = [
        w for w in wallet_manager.wallets if any(token in ["USDT", "USDC", "DAI"] for token in w["tokens"])
    ]
    return jsonify({"stable_wallets": stable_wallets})

@app.route("/wallets/switch-asset", methods=["POST"])
async def switch_asset():
    """
    Swap stable coins to volatile assets or vice versa based on AI-driven decision.
    Useful for scenarios where NAV doubling targets need flexible asset management.
    """
    data = request.json
    wallet_id = data["wallet_id"]
    target_asset = data["target_asset"]
    wallet = next((w for w in wallet_manager.wallets if w["wallet_id"] == wallet_id), None)
    
    if not wallet:
        return jsonify({"error": "Wallet not found"}), 404

    # Perform asset swap using WalletManager's aggregator integration
    success = await wallet_manager.perform_rebalance(wallet)
    
    if success:
        return jsonify({"status": f"Switched assets in {wallet_id} to {target_asset}"})
    return jsonify({"error": "Asset switch failed"}), 500

if __name__ == "__main__":
    app.run(port=5000)
