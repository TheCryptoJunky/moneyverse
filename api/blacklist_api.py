from flask import Flask, jsonify
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

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

@app.route('/api/blacklist', methods=['GET'])
def get_blacklist():
    """Fetches the entire blacklist from the MySQL database."""
    connection = connect_db()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT token_address, reason, date_added FROM blacklist")
            blacklist = cursor.fetchall()
            return jsonify(blacklist)
        except Error as e:
            return jsonify({"error": f"Error fetching blacklist: {e}"}), 500
        finally:
            connection.close()
    return jsonify({"error": "Could not connect to database"}), 500

if __name__ == "__main__":
    app.run(debug=True)
