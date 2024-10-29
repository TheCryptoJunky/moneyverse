# Full file path: moneyverse/flask_gui/app.py

from flask import Flask, jsonify, request, render_template, redirect, url_for, flash
from managers.configuration_manager import ConfigurationManager
from managers.wallet_manager import WalletManager
from database.security_manager import SecurityManager, UserAuthManager
from database.async_db_handler import AsyncDBHandler
from utils.mempool_analysis import MempoolAnalysis
from utils.nav_monitor import NAVMonitor
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
import base64
import asyncio
import os
import smtplib
from twilio.rest import Client
import requests

# Initialize Flask app and utility objects
app = Flask(__name__)
wallet_manager = WalletManager()
config_manager = ConfigurationManager()
security_manager = SecurityManager()
auth_manager = UserAuthManager()
db_handler = AsyncDBHandler()
mempool_analysis = MempoolAnalysis(web3_provider="YOUR_WEB3_PROVIDER_URL")
nav_monitor = NAVMonitor(...)  # Initialize with necessary dependencies

# Centralized encryption and logging setup
transaction_stats = {
    "most_expensive": {"time": None, "aggregator": None, "cost": 0},
    "least_expensive": {"time": None, "aggregator": None, "cost": float('inf')}
}

# --- 2FA Setup ---
@app.route('/setup_2fa', methods=['GET', 'POST'])
def setup_2fa():
    """Sets up 2FA for user authentication with QR code."""
    if request.method == 'POST':
        user_identifier = request.form['username']
        auth_manager.setup_2fa(user_identifier)
        flash("2FA setup complete. Scan the QR code in your authenticator app.")
    return render_template('setup_2fa.html')

@app.route('/validate_2fa', methods=['POST'])
def validate_2fa():
    """Validates the 2FA token to authorize user actions."""
    token = request.form['token']
    if auth_manager.verify_2fa(token):
        return redirect(url_for('wallet_management'))
    else:
        flash("Invalid 2FA token. Please try again.")
        return redirect(url_for('setup_2fa'))

# --- Configuration Endpoints ---
@app.route("/config/<key>", methods=["GET", "POST", "DELETE"])
def config_operations(key):
    """Handles getting, setting, and deleting configuration values."""
    if request.method == "GET":
        value = config_manager.get_config(key)
        return jsonify({key: value}) if value else jsonify({"error": f"Configuration for {key} not found"}), 404
    elif request.method == "POST":
        value = request.json.get("value")
        if value is None:
            return jsonify({"error": "Missing configuration value"}), 400
        config_manager.set_config(key, value)
        return jsonify({"status": f"Configuration for {key} set to {value}."})
    elif request.method == "DELETE":
        config_manager.delete_config(key)
        return jsonify({"status": f"Configuration for {key} deleted."})

# --- Wallet Profile and Management Endpoints ---
@app.route("/wallets/<wallet_id>/profile", methods=["GET"])
async def wallet_profile(wallet_id):
    """Retrieve the full profile for a specific wallet with encrypted data."""
    wallet = await db_handler.fetch("SELECT * FROM wallets WHERE wallet_id = $1", wallet_id)
    if not wallet:
        return jsonify({"error": "Wallet not found"}), 404

    wallet_data = wallet[0]  # Assuming single result for unique wallet_id
    wallet_profile = {
        "wallet_id": wallet_data['wallet_id'],
        "created_date": wallet_data['creation_date'].strftime('%Y-%m-%d'),
        "entry_type": "swarm-born" if wallet_data['wallet_id'].startswith("sw") else "adopted",
        "entry_date": wallet_data['creation_date'].strftime('%Y-%m-%d'),
        "initial_assets": wallet_data['balance'],
        "performance": {
            "overall_gain": 1200,  # Example gain
            "successful_strategies": ["Arbitrage", "Market Making"],
            "mev_techniques": {"sandwich": 30, "backrunning": 50, "arbitrage": 20}
        },
        "asset_distribution": {"USDC": 300, "ETH": 150, "DAI": 50}
    }
    return jsonify(wallet_profile)

@app.route("/wallets/<wallet_id>/details", methods=["GET"])
async def wallet_details(wallet_id):
    """Displays detailed wallet info, including decrypted recovery phrase with 2FA validation."""
    otp_token = request.args.get("otp_token")
    if not auth_manager.verify_2fa(otp_token):
        return jsonify({"error": "Invalid or missing OTP token"}), 403

    wallet = await db_handler.fetch("SELECT * FROM wallets WHERE wallet_id = $1", wallet_id)
    if not wallet:
        return jsonify({"error": "Wallet not found"}), 404

    # Decrypt sensitive recovery phrase
    encrypted_phrase = wallet[0].get("encrypted_seed_phrase", None)
    decrypted_phrase = security_manager.decrypt_data(encrypted_phrase) if encrypted_phrase else "N/A"
    asset_distribution = wallet[0].get("asset_distribution", {})
    pie_img = generate_pie_chart(asset_distribution, title="Asset Distribution")

    details = {
        "wallet_id": wallet_id,
        "address": wallet[0].get("address"),
        "recovery_phrase": decrypted_phrase,
        "assets": wallet[0].get("assets"),
        "asset_distribution_chart": pie_img
    }
    return jsonify(details)

@app.route("/wallets/<wallet_id>/deactivate", methods=["POST"])
async def deactivate_wallet(wallet_id):
    """Deactivates a wallet from swarm activity without deletion."""
    await db_handler.execute("UPDATE wallets SET status = $1 WHERE wallet_id = $2", "inactive", wallet_id)
    return jsonify({"status": f"Wallet {wallet_id} deactivated from swarm activity."})

@app.route("/wallets/<wallet_id>/secure_recovery_phrase", methods=["POST"])
async def secure_recovery_phrase(wallet_id):
    """Encrypts and securely stores the recovery phrase for a wallet."""
    recovery_phrase = request.json.get("recovery_phrase")
    if not recovery_phrase:
        return jsonify({"error": "Recovery phrase is required."}), 400

    encrypted_phrase = security_manager.encrypt_data(recovery_phrase)
    await db_handler.execute("UPDATE wallets SET encrypted_seed_phrase = $1 WHERE wallet_id = $2", encrypted_phrase, wallet_id)
    return jsonify({"status": f"Recovery phrase for wallet {wallet_id} securely stored."})

# --- Mempool Analysis: Data Collection and Export Endpoints ---
@app.route("/toggle_data_collection", methods=["POST"])
def toggle_data_collection():
    """Toggle data collection mode and set interval type and value for analysis."""
    data = request.json
    collect_mode = data.get("collect_mode", False)
    interval_type = data.get("interval_type", "time")
    interval_value = data.get("interval_value", 60)
    
    asyncio.create_task(mempool_analysis.run_analysis_loop(collect_mode, interval_type, interval_value))
    return jsonify({"status": "Data collection toggled.", "collect_mode": collect_mode})

@app.route("/export_data", methods=["POST"])
def export_data():
    """Export historical mempool data within a time range to CSV."""
    data = request.json
    mempool_analysis.export_data(data.get("start_time"), data.get("end_time"))
    return jsonify({"status": "Data export initiated"})

# Utility for Generating Pie Charts
def generate_pie_chart(data, title="Chart"):
    """Generate a base64-encoded pie chart image from data."""
    fig, ax = plt.subplots()
    ax.pie(list(data.values()), labels=list(data.keys()), autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    plt.title(title)
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

if __name__ == "__main__":
    asyncio.run(db_handler.init_pool())
    app.run(port=5000)
