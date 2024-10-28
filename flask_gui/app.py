from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import jwt
import datetime
import logging

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # Ensure this is securely set in the environment

# Sample user data for authentication (consider replacing with a database for production)
USERS = {
    "admin": "password123"  # WARNING: Insecure, replace with proper hashed passwords
}

# Decorator to require a valid JWT token for accessing certain routes
def token_required(f):
    """A decorator to check if the user is authenticated via JWT."""
    def wrap(*args, **kwargs):
        token = session.get('token', None)
        if not token:
            logging.warning("Unauthorized access attempt without token.")
            return redirect(url_for('login'))

        try:
            jwt.decode(token, app.secret_key, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            logging.error("Token expired.")
            return redirect(url_for('login'))
        except jwt.InvalidTokenError:
            logging.error("Invalid token.")
            return redirect(url_for('login'))

        return f(*args, **kwargs)
    return wrap

@app.route('/')
@token_required
def index():
    """Display the home page with environment variables."""
    api_key_1 = os.getenv('API_KEY_1', '')
    secret_key_1 = os.getenv('SECRET_KEY_1', '')
    return render_template('index.html', api_key_1=api_key_1, secret_key_1=secret_key_1)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login and JWT token generation."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if USERS.get(username) == password:
            # Create JWT token with a 30-minute expiration
            token = jwt.encode({
                'user': username,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
            }, app.secret_key, algorithm="HS256")
            session['token'] = token
            logging.info(f"User {username} logged in successfully.")
            return redirect(url_for('index'))
        else:
            logging.warning(f"Failed login attempt for user {username}.")
            return "Invalid credentials", 401
    return render_template('login.html')

@app.route('/save_env', methods=['POST'])
@token_required
def save_env():
    """Save new API keys and secrets to the .env file."""
    api_key_1 = request.form.get('api_key_1', '')
    secret_key_1 = request.form.get('secret_key_1', '')

    # Simple validation for the form inputs (you can extend this)
    if not api_key_1 or not secret_key_1:
        logging.error("API Key or Secret Key is missing.")
        return "API Key and Secret Key are required", 400

    # Save the keys to the .env file
    try:
        with open('.env', 'w') as f:
            f.write(f"API_KEY_1={api_key_1}\n")
            f.write(f"SECRET_KEY_1={secret_key_1}\n")
        logging.info("API keys saved successfully.")
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Error saving .env file: {e}")
        return "Error saving environment variables", 500

if __name__ == "__main__":
    app.run(debug=True)
