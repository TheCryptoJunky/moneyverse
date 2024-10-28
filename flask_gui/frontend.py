from flask import Flask, render_template, request, jsonify
from src.managers.arbitrage_manager import ArbitrageManager  # Assuming a manager to handle the bot instances

app = Flask(__name__)
manager = ArbitrageManager()  # Instantiate a manager to control the bots

@app.route('/')
def home():
    """Render the control panel."""
    return render_template('index.html')

@app.route('/start_bot', methods=['POST'])
def start_bot():
    """Start a specified arbitrage bot."""
    bot_name = request.json['bot']
    manager.start_bot(bot_name)
    return jsonify({"status": "started", "bot": bot_name})

@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    """Stop a specified arbitrage bot."""
    bot_name = request.json['bot']
    manager.stop_bot(bot_name)
    return jsonify({"status": "stopped", "bot": bot_name})

@app.route('/pause_bot', methods=['POST'])
def pause_bot():
    """Pause a specified arbitrage bot."""
    bot_name = request.json['bot']
    manager.pause_bot(bot_name)
    return jsonify({"status": "paused", "bot": bot_name})

@app.route('/update_config', methods=['POST'])
def update_config():
    """Update the configuration for a specific bot."""
    bot_name = request.json['bot']
    config_value = request.json['config']
    manager.update_config(bot_name, config_value)
    return jsonify({"status": "config_updated", "bot": bot_name})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
