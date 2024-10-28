import jwt
import datetime
from flask import jsonify, session
import os
import all_logging

# Secret key should be securely loaded from the environment
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")  # This should be securely set in production

def generate_token(username):
    """
    Generate JWT token with expiration time for a user.
    The token is valid for 30 minutes.
    """
    expiration = datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
    token = jwt.encode({'user': username, 'exp': expiration}, SECRET_KEY, algorithm="HS256")
    logging.info(f"Generated JWT token for user: {username}")
    return token

def verify_token(token):
    """
    Verify JWT token. If the token is expired or invalid, return None.
    """
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return data
    except jwt.ExpiredSignatureError:
        logging.warning("JWT token expired.")
        return None  # Token expired
    except jwt.InvalidTokenError:
        logging.warning("Invalid JWT token.")
        return None  # Invalid token

def login_user(username):
    """
    Log the user in by generating a JWT token and saving it in the session.
    """
    token = generate_token(username)
    session['token'] = token
    logging.info(f"User {username} logged in successfully.")
    return jsonify({'message': 'Logged in successfully', 'token': token})
