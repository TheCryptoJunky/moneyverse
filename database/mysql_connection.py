# File: src/database/mysql_connection.py

import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables from .env file located in project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

def create_connection():
    """
    Establishes a connection to the MySQL database using credentials from the .env file.
    Returns a MySQL connection object or None if the connection fails.
    """
    try:
        # Fetch MySQL credentials from environment variables
        mysql_host = os.getenv('MYSQL_HOST')
        mysql_user = os.getenv('MYSQL_USER')
        mysql_password = os.getenv('MYSQL_PASSWORD')
        mysql_database = os.getenv('MYSQL_DATABASE')

        # Log the MySQL credentials being used (excluding password for security)
        print(f"Connecting to MySQL database '{mysql_database}' at host '{mysql_host}' as user '{mysql_user}'")

        # Establish the MySQL connection
        connection = mysql.connector.connect(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database
        )

        if connection.is_connected():
            db_info = connection.get_server_info()
            print(f"Connected to MySQL Server version {db_info}")
            return connection
        else:
            print("Failed to connect to MySQL Server")

    except Error as e:
        print(f"Error while connecting to MySQL: {str(e)}")
        return None

def test_connection():
    """
    Test the MySQL connection and fetches the current database name.
    """
    connection = create_connection()

    if connection:
        try:
            # Create a cursor to perform SQL operations
            cursor = connection.cursor()
            # Execute a query to get the current database
            cursor.execute("SELECT DATABASE();")
            record = cursor.fetchone()
            print(f"You're connected to database: {record}")

        except Error as e:
            print(f"Error while querying the database: {str(e)}")

        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
                print("MySQL connection is closed")

if __name__ == "__main__":
    test_connection()
