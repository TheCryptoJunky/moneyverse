# File: flask_gui/app.py

from flask import Flask, render_template, request, jsonify
from wallet.wallet_swarm import WalletSwarm
from strategies.mev_strategy import MEV_STRATEGIES

app = Flask(__name__)

# Initialize the wallet swarm with some placeholder addresses
wallet_swarm = WalletSwarm(mev_strategy=None, wallet_addresses=["0xWalletAddress1", "0xWalletAddress2"])

@app.route('/')
def index():
    return render_template('index.html', total_nav=wallet_swarm.calculate_total_net_value(), wallets=wallet_swarm.wallets)

@app.route('/wallet/<address>', methods=['GET'])
def get_wallet_details(address):
    """
    Retrieve and return detailed information for a specific wallet.
    """
    wallet = next((w for w in wallet_swarm.wallets if w.address == address), None)
    if wallet:
        return jsonify({"address": wallet.address, "balance": wallet.balance, "assets": wallet.assets}), 200
    return jsonify({"error": "Wallet not found"}), 404

@app.route('/add_wallet', methods=['POST'])
def add_wallet():
    """
    Add a new wallet to the swarm.
    """
    address = request.form.get('address')
    initial_balance = float(request.form.get('initial_balance', 0))
    wallet_swarm.add_wallet(address=address, initial_balance=initial_balance)
    return jsonify({"status": "Wallet added"}), 200

@app.route('/remove_wallet', methods=['POST'])
def remove_wallet():
    """
    Remove an existing wallet from the swarm.
    """
    address = request.form.get('address')
    wallet_swarm.remove_wallet(address)
    return jsonify({"status": "Wallet removed"}), 200

@app.route('/transfer', methods=['POST'])
def transfer_assets():
    """
    Transfer assets between wallets or to an external address.
    """
    from_address = request.form.get('from_address')
    to_address = request.form.get('to_address')
    asset = request.form.get('asset')
    amount = float(request.form.get('amount'))
    success = wallet_swarm.transfer_asset(from_address, to_address, asset, amount)
    if success:
        return jsonify({"status": "Transfer successful"}), 200
    return jsonify({"error": "Transfer failed"}), 400

@app.route('/swap_assets', methods=['POST'])
def swap_assets():
    """
    Swap assets within a wallet using an aggregator.
    """
    address = request.form.get('address')
    from_asset = request.form.get('from_asset')
    to_asset = request.form.get('to_asset')
    amount = float(request.form.get('amount'))
    success = wallet_swarm.swap_asset(address, from_asset, to_asset, amount)
    if success:
        return jsonify({"status": "Swap successful"}), 200
    return jsonify({"error": "Swap failed"}), 400

if __name__ == "__main__":
    app.run(debug=True)
