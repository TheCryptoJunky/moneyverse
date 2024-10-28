from flask import Flask, render_template, request, redirect, url_for, jsonify
from wallet.wallet_swarm import WalletSwarm
from strategies.mev_strategy import MEV_STRATEGIES
import time

app = Flask(__name__)

wallet_swarm = WalletSwarm(mev_strategy=None, wallet_addresses=["0xWalletAddress1", "0xWalletAddress2"])
historical_nav = []

@app.route('/')
def index():
    return render_template('index.html', total_nav=wallet_swarm.calculate_total_net_value(), wallets=wallet_swarm.wallets)

@app.route('/start_strategy', methods=['POST'])
def start_strategy():
    strategy_name = request.form.get('strategy')
    if strategy_name in MEV_STRATEGIES:
        strategy_class = MEV_STRATEGIES[strategy_name]
        wallet_swarm.mev_strategy = strategy_class(wallet_swarm)
        return jsonify({"status": f"{strategy_name} strategy started"}), 200
    return jsonify({"error": "Invalid strategy"}), 400

@app.route('/stop_strategy', methods=['POST'])
def stop_strategy():
    wallet_swarm.mev_strategy = None
    return jsonify({"status": "Strategy stopped"}), 200

@app.route('/swarm_status')
def swarm_status():
    nav = wallet_swarm.calculate_total_net_value()
    historical_nav.append((time.time(), nav))
    return jsonify({
        "total_nav": nav,
        "wallets": [{"address": w.address, "balance": w.balance} for w in wallet_swarm.wallets]
    })

@app.route('/nav_history')
def nav_history():
    return jsonify(historical_nav)

if __name__ == "__main__":
    app.run(debug=True)
