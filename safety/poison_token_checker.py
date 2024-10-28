# File: /src/safety/poison_token_checker.py

import logging
import requests
from mysql.connector import Error
from src.ai.ai_helpers import TokenSafetyHelper  # AI Helper for dynamic token safety checks

# Initialize logger
logger = logging.getLogger(__name__)

class PoisonTokenChecker:
    """
    Poison Token Checker that dynamically checks tokens against a blacklist and external APIs to ensure safety.
    Incorporates AI-driven logic to detect malicious tokens in real time and prevent interaction with scam tokens.
    """

    def __init__(self, token_api_url, mysql_config):
        """
        Initializes the Poison Token Checker with external API and MySQL configuration for blacklist management.
        """
        self.token_api_url = token_api_url
        self.mysql_config = mysql_config
        self.token_helper = TokenSafetyHelper()  # AI Helper for real-time token safety checks

        # Connect to the MySQL database
        self.connection = self.connect_db()

    def connect_db(self):
        """
        Connects to the MySQL database using the provided configuration for blacklist/whitelist management.
        """
        try:
            return mysql.connector.connect(
                host=self.mysql_config['host'],
                user=self.mysql_config['user'],
                password=self.mysql_config['password'],
                database=self.mysql_config['database']
            )
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise

    def is_token_safe(self, token_address):
        """
        Checks if a token is safe to interact with, leveraging AI-driven checks and external API results.
        """
        if self.is_blacklisted(token_address):
            logger.warning(f"Token {token_address} is blacklisted.")
            return False

        if self.is_whitelisted(token_address):
            logger.info(f"Token {token_address} is whitelisted.")
            return True

        return self.check_external_api(token_address)

    def is_blacklisted(self, token_address):
        """
        Checks if the token is in the local blacklist stored in the database.
        """
        try:
            cursor = self.connection.cursor()
            query = "SELECT token_address FROM blacklist WHERE token_address = %s"
            cursor.execute(query, (token_address,))
            result = cursor.fetchone()
            return result is not None
        except Error as e:
            logger.error(f"Error checking blacklist: {e}")
            return False

    def is_whitelisted(self, token_address):
        """
        Checks if the token is in the whitelist stored in the database.
        """
        try:
            cursor = self.connection.cursor()
            query = "SELECT token_address FROM whitelist WHERE token_address = %s"
            cursor.execute(query, (token_address,))
            result = cursor.fetchone()
            return result is not None
        except Error as e:
            logger.error(f"Error checking whitelist: {e}")
            return False

    def add_to_blacklist(self, token_address):
        """
        Adds a malicious token to the blacklist.
        """
        try:
            cursor = self.connection.cursor()
            query = "INSERT INTO blacklist (token_address) VALUES (%s)"
            cursor.execute(query, (token_address,))
            self.connection.commit()
            logger.info(f"Token {token_address} added to blacklist.")
        except Error as e:
            logger.error(f"Error adding token {token_address} to blacklist: {e}")

    def check_external_api(self, token_address):
        """
        Checks the token using an external API to verify if it's malicious. AI Helper enhances API results.
        """
        try:
            response = requests.get(f"{self.token_api_url}/{token_address}")
            data = response.json()

            # AI-driven analysis of the API response
            if self.token_helper.is_token_scam(data):
                logger.warning(f"Token {token_address} identified as a scam.")
                self.add_to_blacklist(token_address)
                return False

            return True
        except Exception as e:
            logger.error(f"Error checking token via external API: {e}")
            return False

    def close_connection(self):
        """
        Closes the MySQL connection when done.
        """
        if self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed.")
