# Full file path: /moneyverse/flask_gui/app.py

from flask import Flask, jsonify, request
from managers.configuration_manager import ConfigurationManager
import pyotp
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet
from io import BytesIO
import base64
from managers.wallet_manager import WalletManager
from datetime import datetime
import json

app = Flask(__name__)
wallet_manager = WalletManager()
config_manager = ConfigurationManager()


# Load or generate encryption key (store securely in production)
encryption_key = Fernet.generate_key()
cipher = Fernet(encryption_key)

# OTP for secure access (configure with user-specific secret for production)
otp_secret = pyotp.random_base32()
totp = pyotp.TOTP(otp_secret)

# Transaction stats storage
transaction_stats = {
    "most_expensive": {"time": None, "aggregator": None, "cost": 0},
    "least_expensive": {"time": None, "aggregator": None, "cost": float('inf')}
}

# --- Configuration Endpoints ---

@app.route("/config/<key>", methods=["GET"])
def get_configuration(key):
    """Retrieve a configuration value by key."""
    value = config_manager.get_config(key)
    if value is None:
        return jsonify({"error": f"Configuration for {key} not found"}), 404
    return jsonify({key: value})

@app.route("/config/<key>", methods=["POST"])
def set_configuration(key):
    """Set a configuration value by key."""
    data = request.json
    value = data.get("value")
    if value is None:
        return jsonify({"error": "Missing configuration value"}), 400
    config_manager.set_config(key, value)
    return jsonify({"status": f"Configuration for {key} set to {value}."})

@app.route("/config/<key>", methods=["DELETE"])
def delete_configuration(key):
    """Delete a configuration by key."""
    config_manager.delete_config(key)
    return jsonify({"status": f"Configuration for {key} deleted."})

# --- Enhanced Wallet Profile Endpoints ---

@app.route("/wallets/<wallet_id>/profile", methods=["GET"])
def wallet_profile(wallet_id):
    """Retrieve the full profile for a specific wallet, including sensitive details."""
    wallet = next((w for w in wallet_manager.wallets if w["wallet_id"] == wallet_id), None)
    if not wallet:
        return jsonify({"error": "Wallet not found"}), 404

    # Mock wallet profile data (extend this with real data sources as needed)
    wallet_profile = {
        "wallet_id": wallet["wallet_id"],
        "created_date": "2024-01-15",
        "entry_type": "swarm-born" if wallet["wallet_id"].startswith("sw") else "adopted",
        "entry_date": "2024-01-20",
        "initial_assets": 5000,
        "performance": {
            "overall_gain": 1200,  # example
            "successful_strategies": ["Arbitrage", "Market Making"],
            "mev_techniques": {
                "sandwich": 30,
                "backrunning": 50,
                "arbitrage": 20
            }
        },
        "asset_distribution": {
            "USDC": 300,
            "ETH": 150,
            "DAI": 50
        }
    }

    return jsonify(wallet_profile)

@app.route("/wallets/<wallet_id>/details", methods=["GET"])
def wallet_details(wallet_id):
    """Show detailed wallet info, including encrypted recovery phrase, secured by OTP."""
    otp_token = request.args.get("otp_token")
    if not totp.verify(otp_token):
        return jsonify({"error": "Invalid or missing OTP token"}), 403

    wallet = next((w for w in wallet_manager.wallets if w["wallet_id"] == wallet_id), None)
    if not wallet:
        return jsonify({"error": "Wallet not found"}), 404

    # Decrypt and show recovery phrase securely
    encrypted_phrase = wallet.get("recovery_phrase", None)
    if encrypted_phrase:
        decrypted_phrase = cipher.decrypt(encrypted_phrase).decode()
    else:
        decrypted_phrase = "Recovery phrase not available."

    # Asset distribution pie chart
    asset_distribution = wallet["asset_distribution"]
    pie_img = generate_pie_chart(asset_distribution, title="Asset Distribution")

    details = {
        "wallet_id": wallet_id,
        "address": wallet.get("address"),
        "recovery_phrase": decrypted_phrase,
        "assets": wallet["assets"],
        "asset_distribution_chart": pie_img
    }

    return jsonify(details)

@app.route("/wallets/<wallet_id>/deactivate", methods=["POST"])
def deactivate_wallet(wallet_id):
    """Deactivate a wallet from swarm activity, without deleting it."""
    wallet = next((w for w in wallet_manager.wallets if w["wallet_id"] == wallet_id), None)
    if not wallet:
        return jsonify({"error": "Wallet not found"}), 404

    wallet["status"] = "inactive"
    return jsonify({"status": f"Wallet {wallet_id} deactivated from swarm activity."})

# --- Utility Functions ---

def generate_pie_chart(data, title="Chart"):
    """Generate a base64-encoded pie chart image from data."""
    fig, ax = plt.subplots()
    labels = list(data.keys())
    sizes = list(data.values())
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    plt.title(title)
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    pie_chart_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return pie_chart_base64

@app.route("/wallets/<wallet_id>/secure_recovery_phrase", methods=["POST"])
def secure_recovery_phrase(wallet_id):
    """
    Securely store and encrypt the recovery phrase for a wallet.
    Expects JSON: {"recovery_phrase": "phrase"}
    """
    data = request.json
    recovery_phrase = data.get("recovery_phrase")
    if not recovery_phrase:
        return jsonify({"error": "Recovery phrase is required."}), 400

    encrypted_phrase = cipher.encrypt(recovery_phrase.encode())
    wallet = next((w for w in wallet_manager.wallets if w["wallet_id"] == wallet_id), None)
    if not wallet:
        return jsonify({"error": "Wallet not found"}), 404

    wallet["recovery_phrase"] = encrypted_phrase
    return jsonify({"status": f"Recovery phrase for wallet {wallet_id} securely stored."})

if __name__ == "__main__":
    app.run(port=5000)
