# File: /src/managers/list_manager.py

import mysql.connector
import all_logging
from mysql.connector import Error

class ListManager:
    """
    Manages the various lists: whitelist, blacklist, redlist, and pumplist.
    Interacts with the MySQL database to retrieve and update the lists.
    """
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = self.connect_db()

    def connect_db(self):
        """
        Connect to the MySQL database using the provided configuration.
        """
        try:
            return mysql.connector.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )
        except Error as e:
            logging.error(f"Error connecting to MySQL: {e}")
            raise

    # --- Whitelist Methods ---
    def is_whitelisted(self, token_address):
        """Check if the token is in the whitelist."""
        return self._check_list('whitelist', token_address)

    def add_to_whitelist(self, token_address, added_by):
        """Add a token to the whitelist."""
        return self._add_to_list('whitelist', token_address, added_by)

    def remove_from_whitelist(self, token_address):
        """Remove a token from the whitelist."""
        return self._remove_from_list('whitelist', token_address)

    # --- Blacklist Methods ---
    def is_blacklisted(self, token_address):
        """Check if the token is in the blacklist."""
        return self._check_list('blacklist', token_address)

    def add_to_blacklist(self, token_address, flagged_by, reason, ai_risk_score=1.00):
        """Add a token to the blacklist."""
        return self._add_to_blacklist(token_address, flagged_by, reason, ai_risk_score)

    def remove_from_blacklist(self, token_address):
        """Remove a token from the blacklist."""
        return self._remove_from_list('blacklist', token_address)

    # --- Redlist Methods ---
    def is_redlisted(self, token_address):
        """Check if the token is in the redlist."""
        return self._check_list('redlist', token_address)

    def add_to_redlist(self, token_address, flagged_by, reason):
        """Add a token to the redlist."""
        return self._add_to_list('redlist', token_address, flagged_by, reason)

    def remove_from_redlist(self, token_address):
        """Remove a token from the redlist."""
        return self._remove_from_list('redlist', token_address)

    # --- Pumplist Methods ---
    def is_on_pumplist(self, token_address):
        """Check if the token is in the pumplist."""
        return self._check_list('pumplist', token_address)

    def add_to_pumplist(self, token_address, focus_duration, added_by):
        """Add a token to the pumplist."""
        try:
            cursor = self.connection.cursor()
            query = "INSERT INTO pumplist (token_address, focus_duration, added_by) VALUES (%s, %s, %s)"
            cursor.execute(query, (token_address, focus_duration, added_by))
            self.connection.commit()
            cursor.close()
            logging.info(f"Token {token_address} added to pumplist.")
        except Error as e:
            logging.error(f"Error adding token to pumplist: {e}")

    def remove_from_pumplist(self, token_address):
        """Remove a token from the pumplist."""
        return self._remove_from_list('pumplist', token_address)

    # --- Utility Methods ---
    def _check_list(self, list_name, token_address):
        """Generic method to check if a token exists in a list."""
        try:
            cursor = self.connection.cursor()
            query = f"SELECT token_address FROM {list_name} WHERE token_address = %s"
            cursor.execute(query, (token_address,))
            result = cursor.fetchone()
            cursor.close()
            return result is not None
        except Error as e:
            logging.error(f"Error checking {list_name}: {e}")
            return False

    def _add_to_list(self, list_name, token_address, added_by, reason=None):
        """Generic method to add a token to a list."""
        try:
            cursor = self.connection.cursor()
            query = f"INSERT INTO {list_name} (token_address, added_by, reason) VALUES (%s, %s, %s)"
            cursor.execute(query, (token_address, added_by, reason))
            self.connection.commit()
            cursor.close()
            logging.info(f"Token {token_address} added to {list_name}.")
        except Error as e:
            logging.error(f"Error adding token to {list_name}: {e}")

    def _add_to_blacklist(self, token_address, flagged_by, reason, ai_risk_score):
        """Specific method to add a token to the blacklist with AI risk scoring."""
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO blacklist (token_address, flagged_by, reason, ai_risk_score)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (token_address, flagged_by, reason, ai_risk_score))
            self.connection.commit()
            cursor.close()
            logging.info(f"Token {token_address} added to blacklist with risk score {ai_risk_score}.")
        except Error as e:
            logging.error(f"Error adding token to blacklist: {e}")

    def _remove_from_list(self, list_name, token_address):
        """Generic method to remove a token from a list."""
        try:
            cursor = self.connection.cursor()
            query = f"DELETE FROM {list_name} WHERE token_address = %s"
            cursor.execute(query, (token_address,))
            self.connection.commit()
            cursor.close()
            logging.info(f"Token {token_address} removed from {list_name}.")
        except Error as e:
            logging.error(f"Error removing token from {list_name}: {e}")
