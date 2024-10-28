# File: /src/flask_gui/gui.py

import tkinter as tk
from tkinter import messagebox
import all_logging
from src.database.database_manager import DatabaseManager
from centralized_logger import CentralizedLogger
from bot_manager import BotManager  # To interact with bots and view their status
from src.utils.error_handler import ErrorHandler  # To view errors in real-time

# Initialize logger and centralized logging
logger = logging.getLogger(__name__)
centralized_logger = CentralizedLogger()

class TradingBotGUI:
    """
    A GUI interface to monitor bot activity, manage Greenlist assets, and view system status.
    Built using Tkinter to allow users to manually interact with the system, view real-time logs, and manage bots.
    """

    def __init__(self, root, db_manager, bot_manager, error_handler):
        self.root = root
        self.db_manager = db_manager
        self.bot_manager = bot_manager
        self.error_handler = error_handler

        self.root.title("Crypto Trading Bot Manager")
        self.setup_gui()

    def setup_gui(self):
        """
        Sets up the GUI layout with buttons and status displays.
        """
        # Add a label for Greenlist management
        tk.Label(self.root, text="Greenlist Management", font=('Helvetica', 14)).grid(row=0, column=0, pady=10)
        
        # Add button to view Greenlist assets
        tk.Button(self.root, text="View Greenlist Assets", command=self.view_greenlist).grid(row=1, column=0)

        # Add button to view real-time trades
        tk.Button(self.root, text="View Active Trades", command=self.view_trades).grid(row=2, column=0)

        # Add button to view error logs
        tk.Button(self.root, text="View Error Logs", command=self.view_error_logs).grid(row=3, column=0)

        # Add button to trigger bot reinvestment manually
        tk.Button(self.root, text="Manual Profit Reinvestment", command=self.manual_reinvestment).grid(row=4, column=0)

    def view_greenlist(self):
        """
        Displays Greenlist assets from the database.
        """
        assets = self.db_manager.get_greenlist_assets()  # Fetch assets from the database
        message = "\n".join([f"Asset: {asset['name']}, Priority: {asset['priority']}" for asset in assets])
        messagebox.showinfo("Greenlist Assets", message)

    def view_trades(self):
        """
        Displays active trades from the centralized logger.
        """
        trades = centralized_logger.get_recent_trades()  # Fetch recent trades
        message = "\n".join([f"Trade: {trade['pair']}, Action: {trade['action']}, Size: {trade['size']}" for trade in trades])
        messagebox.showinfo("Active Trades", message)

    def view_error_logs(self):
        """
        Displays error logs from the error handler.
        """
        errors = self.error_handler.get_recent_errors()  # Fetch recent error logs
        message = "\n".join([f"Error: {error['message']}, Time: {error['time']}" for error in errors])
        messagebox.showinfo("Error Logs", message)

    def manual_reinvestment(self):
        """
        Allows the user to manually trigger profit reinvestment across top-performing bots.
        """
        try:
            self.bot_manager.reinvest_profits()  # Trigger reinvestment
            messagebox.showinfo("Success", "Profit reinvestment executed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reinvest profits: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    db_manager = DatabaseManager(db_config={"host": "localhost", "user": "root", "password": "password", "database": "crypto_db"})
    bot_manager = BotManager()
    error_handler = ErrorHandler()
    gui = TradingBotGUI(root, db_manager, bot_manager, error_handler)
    root.mainloop()

# gui.py
import all_logging
import tkinter as tk

class GUI:
    def __init__(self, config):
        self.config = config
        self.root = tk.Tk()

    def run(self):
        self.root.mainloop()

# Usage:
gui = GUI(config)
gui.run()
