# Full file path: /moneyverse/flask_gui/aggressive_dca_dashboard.py

from flask import Flask, jsonify, request
from wallet_swarm import WalletSwarm
from aggressive_dca_bot import ReinforcedAccumulationBot
from ai.ml_refinement import MLRefinementModel
from utils import sentiment_analysis, volatility_metrics
from centralized_logger import CentralizedLogger
from database_manager import update_nav, track_trade_execution, fetch_performance_data

app = Flask(__name__)
logger = CentralizedLogger()
wallet_swarm = WalletSwarm()
accumulation_bot = None
ml_refinement_model = MLRefinementModel(environment="swarm_performance")

# Endpoint to set an accumulation target with a multi-strategy swarm
@app.route("/set_accumulation_target", methods=["POST"])
def set_accumulation_target():
    data = request.get_json()
    target_token = data.get("token_address")  # Target token address for accumulation
    nav_goal = data.get("nav_goal", 2.0)  # NAV multiplier target per hour
    accumulation_threshold = data.get("accumulation_threshold", 500)  # Desired amount of target asset
    time_limit = data.get("time_limit")  # Optional time limit for focused accumulation
    strategies = data.get("strategies", ["front_running", "sandwich"])  # Multi-strategy options
    
    # Initialize the accumulation bot for the target token
    global accumulation_bot
    accumulation_bot = ReinforcedAccumulationBot(target_token, nav_goal, accumulation_threshold, strategies)
    
    # Configure wallet swarm to focus on the target token with the specified strategies
    wallet_swarm.configure(target_token=target_token, strategies=strategies, nav_goal=nav_goal)
    logger.log_info(f"Accumulation target set for {target_token} with NAV goal {nav_goal}x per hour.")

    return jsonify({
        "status": "Accumulation target set",
        "target_token": target_token,
        "nav_goal": nav_goal,
        "accumulation_threshold": accumulation_threshold
    })

# Endpoint to monitor real-time NAV and performance analytics
@app.route("/performance_analytics", methods=["GET"])
def performance_analytics():
    performance_data = fetch_performance_data()  # Fetches NAV, efficiency, and accumulation data from the DB
    logger.log_info("Performance analytics data fetched.")
    return jsonify(performance_data)

# Endpoint to adjust focus for NAV growth after achieving accumulation targets
@app.route("/adjust_focus", methods=["POST"])
def adjust_focus():
    data = request.get_json()
    target_token = data.get("token_address")
    nav_current = data.get("nav_current")

    # Switch focus based on whether the accumulation threshold is met
    if accumulation_bot.check_threshold_reached():
        logger.log_info("Accumulation threshold reached. Switching focus to NAV growth.")
        wallet_swarm.adjust_for_nav_focus(nav_goal=nav_current * 2.0)  # Set to double NAV
        return jsonify({
            "status": "Switched to NAV doubling",
            "nav_target": nav_current * 2.0
        })
    else:
        logger.log_info("Continuing accumulation.")
        return jsonify({"status": "Accumulation in progress"})

# Endpoint to dynamically update NAV goals based on user input
@app.route("/dynamic_nav_goal", methods=["POST"])
def dynamic_nav_goal():
    data = request.get_json()
    nav_goal = data.get("nav_goal")
    wallet_swarm.update_nav_goal(nav_goal=nav_goal)
    logger.log_info(f"NAV goal updated to {nav_goal}x per hour.")
    return jsonify({"status": "NAV goal updated", "nav_goal": nav_goal})

# Endpoint to execute swarm strategies and track performance
@app.route("/execute_swarm", methods=["POST"])
def execute_swarm():
    data = request.get_json()
    target_token = data.get("token_address")
    
    # Execute strategies and track success
    success = wallet_swarm.execute_multi_strategy(target_token=target_token)
    if success:
        track_trade_execution(target_token)  # Log trade success
        return jsonify({"status": "Swarm strategy executed successfully"})
    else:
        logger.log_error("Swarm strategy execution failed.")
        return jsonify({"status": "Swarm strategy execution failed"})

# Endpoint for real-time risk control adjustments based on market conditions
@app.route("/set_risk_controls", methods=["POST"])
def set_risk_controls():
    data = request.get_json()
    risk_tolerance = data.get("risk_tolerance")
    volatility_threshold = data.get("volatility_threshold")
    
    # Update risk settings for wallet swarm
    wallet_swarm.update_risk_controls(risk_tolerance=risk_tolerance, volatility_threshold=volatility_threshold)
    logger.log_info("Risk controls updated dynamically.")
    return jsonify({"status": "Risk controls updated", "risk_tolerance": risk_tolerance, "volatility_threshold": volatility_threshold})

# Endpoint to apply machine learning refinements to strategy configurations
@app.route("/refine_strategy", methods=["POST"])
def refine_strategy():
    data = request.get_json()
    primary_token = data.get("primary_token")
    
    # Use ML model to refine and optimize strategies based on past performance
    refined_strategies = ml_refinement_model.optimize_strategies(wallet_swarm.current_strategies)
    wallet_swarm.update_strategies(refined_strategies)
    logger.log_info(f"Strategies refined by ML for {primary_token}.")
    
    return jsonify({"status": "Strategies refined with ML", "primary_token": primary_token})

if __name__ == "__main__":
    app.run(debug=False, port=5000)
