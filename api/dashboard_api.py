from flask import Flask, jsonify, request, redirect, url_for, render_template
from flask_login import LoginManager, login_required, login_user, logout_user, current_user, UserMixin
import mysql.connector
from mysql.connector import Error
import subprocess  # To control bot processes
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Use a strong, secure key in production

# Flask-Login setup for security
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Dummy user authentication for simplicity
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Mocked users (could connect this to your database for more secure user management)
users = {'admin': 'password'}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            return "Invalid username or password", 401
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# MySQL Database Connection Function
def connect_db():
    """Establishes a connection to the MySQL database."""
    try:
        connection = mysql.connector.connect(
            host="your_host",
            user="your_user",
            password="your_password",
            database="your_db"
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# Bot Controls (Start, Stop, etc.)
@app.route('/api/bot/start/<string:bot_name>', methods=['POST'])
@login_required
def start_bot(bot_name):
    """Starts a specified bot by running the script in the background."""
    try:
        subprocess.Popen(["python", f"bots/{bot_name}.py"])
        return jsonify({"status": "success", "message": f"Bot {bot_name} started"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/bot/stop/<string:bot_name>', methods=['POST'])
@login_required
def stop_bot(bot_name):
    """Stops a specified bot."""
    try:
        # Implement logic to stop a bot, possibly using process control to kill the bot script
        os.system(f"pkill -f {bot_name}.py")
        return jsonify({"status": "success", "message": f"Bot {bot_name} stopped"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Blacklist and Whitelist Management
@app.route('/api/blacklist/add', methods=['POST'])
@login_required
def add_blacklist():
    """Adds a token address to the blacklist."""
    token_address = request.json.get('token_address')
    reason = request.json.get('reason')
    connection = connect_db()
    if connection:
        try:
            cursor = connection.cursor()
            query = "INSERT INTO blacklist (token_address, reason) VALUES (%s, %s)"
            cursor.execute(query, (token_address, reason))
            connection.commit()
            return jsonify({"status": "success", "message": "Token added to blacklist"}), 200
        except Error as e:
            return jsonify({"status": "error", "message": str(e)}), 500
        finally:
            connection.close()
    return jsonify({"error": "Could not connect to database"}), 500

@app.route('/api/blacklist/remove', methods=['POST'])
@login_required
def remove_blacklist():
    """Removes a token address from the blacklist."""
    token_address = request.json.get('token_address')
    connection = connect_db()
    if connection:
        try:
            cursor = connection.cursor()
            query = "DELETE FROM blacklist WHERE token_address = %s"
            cursor.execute(query, (token_address,))
            connection.commit()
            return jsonify({"status": "success", "message": "Token removed from blacklist"}), 200
        except Error as e:
            return jsonify({"status": "error", "message": str(e)}), 500
        finally:
            connection.close()
    return jsonify({"error": "Could not connect to database"}), 500

@app.route('/api/whitelist/add', methods=['POST'])
@login_required
def add_whitelist():
    """Adds a wallet address to the whitelist."""
    wallet_address = request.json.get('wallet_address')
    connection = connect_db()
    if connection:
        try:
            cursor = connection.cursor()
            query = "INSERT INTO whitelist (wallet_address) VALUES (%s)"
            cursor.execute(query, (wallet_address,))
            connection.commit()
            return jsonify({"status": "success", "message": "Address added to whitelist"}), 200
        except Error as e:
            return jsonify({"status": "error", "message": str(e)}), 500
        finally:
            connection.close()
    return jsonify({"error": "Could not connect to database"}), 500

@app.route('/api/whitelist/remove', methods=['POST'])
@login_required
def remove_whitelist():
    """Removes a wallet address from the whitelist."""
    wallet_address = request.json.get('wallet_address')
    connection = connect_db()
    if connection:
        try:
            cursor = connection.cursor()
            query = "DELETE FROM whitelist WHERE wallet_address = %s"
            cursor.execute(query, (wallet_address,))
            connection.commit()
            return jsonify({"status": "success", "message": "Address removed from whitelist"}), 200
        except Error as e:
            return jsonify({"status": "error", "message": str(e)}), 500
        finally:
            connection.close()
    return jsonify({"error": "Could not connect to database"}), 500

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

if __name__ == "__main__":
    app.run(ssl_context='adhoc', debug=True)
