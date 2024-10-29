# Full file path: /moneyverse/flask_gui/dashboard.py

import asyncio
import os
from flask import Flask, jsonify, request, session, redirect, url_for
from flask_login import LoginManager, login_required, UserMixin
from wallet_swarm import WalletSwarm
from managers.database_manager import (
    add_pumplist_entry, remove_pumplist_entry,
    add_redlist_entry, fetch_performance_data,
    load_pumplist_from_db, load_redlist_from_db
)
from strategies.protection import activate_partner_protection
from ai.ml_refinement import MLRefinementModel
from all_logging.centralized_logger import CentralizedLogger
from two_factor_auth import verify_2fa  # Assume this is a 2FA verification function

# Initialize Flask app and components
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
login_manager = LoginManager()
login_manager.init_app(app)
logger = CentralizedLogger()
wallet_swarm = WalletSwarm()
ml_refinement_model = MLRefinementModel(environment="swarm_performance")

# Initialize Lists with asynchronous data loading
pumplist, redlist = [], []

# Load user management
class User(UserMixin):
    def __init__(self, id, role):
        self.id = id
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    return User(user_id, "admin")  # Replace with real role fetch from DB

def role_required(role):
    """Decorator for role-based access to sensitive endpoints."""
    def decorator(func):
        @login_required
        def wrapper(*args, **kwargs):
            if session.get("user_role") == role:
                return func(*args, **kwargs)
            else:
                return jsonify({"status": "Unauthorized"}), 403
        return wrapper
    return decorator

# Initialize database lists
async def sync_lists_from_db():
    global pumplist, redlist
    pumplist = await load_pumplist_from_db()
    redlist = await load_redlist_from_db()
    logger.log_info("Synchronized pumplist and redlist from database.")

asyncio.run(sync_lists_from_db())

# --- Pumplist and Redlist Management Endpoints ---
@app.route("/pumplist/add", methods=["POST"])
def add_to_pumplist():
    """Adds an entry to the Pumplist with specified focus duration and strategies."""
    try:
        data = request.get_json()
        entry = data["entry"]
        duration = data["duration"]
        strategies = data.get("strategies", ["market_maker", "accumulation", "protection"])
        
        pumplist.append({"entry": entry, "duration": duration, "strategies": strategies})
        asyncio.run(add_pumplist_entry(entry, duration, strategies))
        wallet_swarm.reallocate_resources(entry, strategies)
        
        logger.log_info(f"Added {entry} to Pumplist with duration {duration} minutes.")
        return jsonify({"status": "Entry added to Pumplist", "entry": entry, "duration": duration})
    except Exception as e:
        logger.log_error(f"Failed to add {entry} to Pumplist: {e}")
        return jsonify({"status": "Error adding entry to Pumplist", "error": str(e)}), 500

@app.route("/redlist/add", methods=["POST"])
def add_to_redlist():
    """Adds a specified actor to the Redlist for ongoing targeting."""
    try:
        data = request.get_json()
        bad_actor = data["bad_actor"]
        
        redlist.append(bad_actor)
        asyncio.run(add_redlist_entry(bad_actor))
        
        logger.log_info(f"Added {bad_actor} to Redlist.")
        return jsonify({"status": "Bad actor added to Redlist", "bad_actor": bad_actor})
    except Exception as e:
        logger.log_error(f"Failed to add {bad_actor} to Redlist: {e}")
        return jsonify({"status": "Error adding bad actor to Redlist", "error": str(e)}), 500

# --- Real-Time Analytics and Reporting ---
@app.route("/performance/analytics", methods=["GET"])
@role_required("admin")  # Restricted to admin users
@login_required
def performance_analytics():
    """Retrieves performance analytics for all active strategies and swarm behavior."""
    try:
        analytics_data = asyncio.run(fetch_performance_data())
        logger.log_info("Retrieved performance analytics.")
        return jsonify({"analytics_data": analytics_data, "message": "Performance analytics retrieved successfully"})
    except Exception as e:
        logger.log_error(f"Error retrieving performance analytics: {e}")
        return jsonify({"status": "Error retrieving performance analytics", "error": str(e)}), 500

# --- Framework Control (Start/Stop) ---
@app.route("/framework/control", methods=["POST"])
@role_required("admin")
@login_required
def control_framework():
    """Controls the framework's operation by starting or stopping the entire system."""
    try:
        action = request.get_json().get("action")
        if action == "start":
            wallet_swarm.start_framework()
            logger.log_info("Framework started.")
            return jsonify({"status": "Framework started"})
        elif action == "stop":
            wallet_swarm.stop_framework()
            logger.log_info("Framework stopped.")
            return jsonify({"status": "Framework stopped"})
        else:
            return jsonify({"status": "Invalid action specified"}), 400
    except Exception as e:
        logger.log_error(f"Error controlling framework: {e}")
        return jsonify({"status": "Error controlling framework", "error": str(e)}), 500

# --- Real-Time Configuration Controls ---
@app.route("/config/update", methods=["POST"])
@role_required("admin")
@login_required
def update_configuration():
    """Updates dynamic configuration settings, such as NAV targets and DCA intervals."""
    try:
        data = request.get_json()
        nav_target = data.get("nav_target", wallet_swarm.nav_target)
        dca_interval = data.get("dca_interval", wallet_swarm.dca_interval)
        
        wallet_swarm.update_nav_target(nav_target)
        wallet_swarm.update_dca_interval(dca_interval)
        
        logger.log_info(f"Updated NAV target to {nav_target} and DCA interval to {dca_interval}")
        return jsonify({"status": "Configuration updated", "nav_target": nav_target, "dca_interval": dca_interval})
    except Exception as e:
        logger.log_error(f"Error updating configuration: {e}")
        return jsonify({"status": "Error updating configuration", "error": str(e)}), 500

# --- Strategy Adjustment and ML Refinement ---
@app.route("/strategies/refine", methods=["POST"])
@role_required("admin")
@login_required
def refine_strategies():
    """Refines current strategies using ML-based optimization to improve swarm performance."""
    try:
        original_strategies = wallet_swarm.current_strategies
        refined_strategies = ml_refinement_model.optimize_strategies(original_strategies)
        wallet_swarm.update_strategies(refined_strategies)
        
        logger.log_info("Strategies refined by ML model.")
        return jsonify({
            "status": "Strategies refined",
            "original_strategies": original_strategies,
            "refined_strategies": refined_strategies
        })
    except Exception as e:
        logger.log_error(f"Error in strategy refinement: {e}")
        return jsonify({"status": "Error refining strategies", "error": str(e)}), 500

# --- 2FA Verification ---
@app.route("/2fa_verify", methods=["POST"])
def verify_2fa_route():
    """Handles 2FA verification for additional security on sensitive dashboards."""
    code = request.form.get("code")
    if verify_2fa(session["user_id"], code):
        session["2fa_verified"] = True
        return redirect(url_for("dashboard"))
    return jsonify({"status": "2FA verification failed"}), 403

@app.route("/sensitive_dashboard", methods=["GET"])
@role_required("admin")
@login_required
def sensitive_dashboard():
    """Restricted access to sensitive dashboard data."""
    if not session.get("2fa_verified"):
        return redirect(url_for("verify_2fa_route"))
    return jsonify({"status": "Access granted to sensitive dashboard"})

# Start the application
if __name__ == "__main__":
    app.run(debug=True, port=5000)
