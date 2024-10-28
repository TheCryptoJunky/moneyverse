# File: /src/database/__init__.py

import mysql.connector

def insert_greenlist_log(wallet_address, action, timestamp):
    """
    Insert a log entry into the Greenlist logs.
    
    :param wallet_address: The address of the wallet being logged.
    :param action: The action performed (e.g., added, removed).
    :param timestamp: The time of the action.
    """
    # Code to insert the log into MySQL database
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="crypto_db"
    )
    cursor = connection.cursor()
    query = "INSERT INTO greenlist_logs (wallet_address, action, timestamp) VALUES (%s, %s, %s)"
    cursor.execute(query, (wallet_address, action, timestamp))
    connection.commit()
    cursor.close()
    connection.close()

def get_greenlist():
    """
    Retrieve the current greenlisted assets.
    
    :return: A list of wallet addresses that are currently greenlisted.
    """
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="crypto_db"
    )
    cursor = connection.cursor()
    cursor.execute("SELECT wallet_address FROM greenlist")
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    
    return [row[0] for row in result]

def add_to_greenlist(wallet_address):
    """
    Add a wallet address to the Greenlist.
    
    :param wallet_address: The wallet address to be added.
    """
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="crypto_db"
    )
    cursor = connection.cursor()
    query = "INSERT INTO greenlist (wallet_address) VALUES (%s)"
    cursor.execute(query, (wallet_address,))
    connection.commit()
    cursor.close()
    connection.close()

def remove_from_greenlist(wallet_address):
    """
    Remove a wallet address from the Greenlist.
    
    :param wallet_address: The wallet address to be removed.
    """
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="crypto_db"
    )
    cursor = connection.cursor()
    query = "DELETE FROM greenlist WHERE wallet_address = %s"
    cursor.execute(query, (wallet_address,))
    connection.commit()
    cursor.close()
    connection.close()
