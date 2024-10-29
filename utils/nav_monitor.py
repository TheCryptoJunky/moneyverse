# Full file path: moneyverse/utils/nav_monitor.py

import pandas as pd
import smtplib
import requests
import os
from datetime import datetime, timedelta
from typing import Dict
from twilio.rest import Client  # For SMS notifications
from .nav_calculator import calculate_nav
from .performance_tracker import PerformanceTracker
from ai.rl_agent import RLTradingAgent
from managers.wallet_manager import WalletManager
from database.security_manager import SecurityManager

class NAVMonitor:
    def __init__(self, performance_tracker: PerformanceTracker, rl_agent: RLTradingAgent, wallet_manager: WalletManager, db_handler, twilio_client=None, goal_multiplier=2.0, alert_threshold=0.2):
        """
        Initialize NAVMonitor to track NAV history, predict trends, and manage alerts and notifications.

        Args:
            performance_tracker (PerformanceTracker): Tracker for NAV performance updates.
            rl_agent (RLTradingAgent): RL agent for AI-based strategy adjustment.
            wallet_manager (WalletManager): Manages wallet transactions.
            db_handler: Database handler for retrieving notification settings.
            twilio_client: Initialized Twilio client for SMS notifications (optional).
            goal_multiplier (float): The target NAV increase per hour (e.g., 2.0 for doubling NAV).
            alert_threshold (float): Threshold for triggering alerts (e.g., 0.2 for 20%).
        """
        self.performance_tracker = performance_tracker
        self.rl_agent = rl_agent
        self.wallet_manager = wallet_manager
        self.db_handler = db_handler
        self.twilio_client = twilio_client
        self.goal_multiplier = goal_multiplier
        self.alert_threshold = alert_threshold
        self.nav_history = pd.DataFrame(columns=['timestamp', 'nav'])
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=1)
        self.profit_wallet = "non_swarm_wallet"  # Specify designated profit wallet ID
        self.last_nav = 0

    def monitor_nav(self, current_timestamp: int, current_nav: float) -> None:
        """
        Updates NAV history, tracks current NAV, and checks for alert conditions.

        Args:
            current_timestamp (int): Timestamp of the current NAV update.
            current_nav (float): Current NAV value.
        """
        self.nav_history = self.nav_history.append({'timestamp': current_timestamp, 'nav': current_nav}, ignore_index=True)
        self.performance_tracker.update_nav_history(self.nav_history)
        self.check_alerts(current_nav)
        self.last_nav = current_nav  # Track last NAV for comparisons

    def check_alerts(self, current_nav: float) -> None:
        """
        Checks if NAV trends meet conditions for alerts and manages profit allocation.

        Args:
            current_nav (float): The current NAV value.
        """
        # Calculate target NAV and surplus threshold
        initial_nav = self.nav_history.iloc[0]['nav']
        target_nav = initial_nav * self.goal_multiplier
        surplus_threshold = target_nav * (1 + self.alert_threshold)

        # Calculate predicted NAV trend
        nav_prediction = self.rl_agent.predict_nav_trend(self.nav_history)

        # Alert for potential loss
        if nav_prediction < current_nav * (1 - self.alert_threshold):
            self.send_notification("ALERT: Potential NAV drop exceeding 20% detected.")
            self.performance_tracker.alert("warning", "Potential NAV drop exceeding 20% detected.")

        # Alert for surplus above target goal
        if nav_prediction > surplus_threshold:
            self.send_notification("SUCCESS: Projected NAV surplus exceeding 20% over doubling goal.")
            self.allocate_surplus(current_nav, target_nav)

    def allocate_surplus(self, current_nav: float, target_nav: float) -> None:
        """
        Allocates surplus profits exceeding the doubling goal by 20% to a separate wallet.

        Args:
            current_nav (float): Current NAV value.
            target_nav (float): Target NAV value for the hour.
        """
        surplus_amount = current_nav - target_nav * (1 + self.alert_threshold)
        if surplus_amount > 0:
            self.wallet_manager.transfer_to_wallet(self.profit_wallet, surplus_amount)
            self.performance_tracker.log("info", f"Transferred surplus of {surplus_amount} to profit wallet.")

    def countdown_status(self) -> Dict:
        """
        Provides real-time countdown status, estimated NAV, and growth or loss percentage.

        Returns:
            Dict: Countdown status with remaining time, projected NAV, and growth/loss percentage.
        """
        remaining_time = (self.end_time - datetime.now()).total_seconds()
        nav_change = ((self.last_nav - self.nav_history.iloc[0]['nav']) / self.nav_history.iloc[0]['nav']) * 100
        estimated_final_nav = self.last_nav + (nav_change * (remaining_time / 3600))

        return {
            "remaining_time": max(remaining_time, 0),
            "estimated_nav": estimated_final_nav,
            "nav_change_percent": nav_change
        }

    async def update_status_on_gui(self, flask_app):
        """
        Periodically updates the countdown and NAV estimate in the Flask GUI.

        Args:
            flask_app (Flask): The Flask app instance for GUI updates.
        """
        while datetime.now() < self.end_time:
            status = self.countdown_status()
            with flask_app.app_context():
                flask_app.update_dashboard(status)  # Assumes `update_dashboard` is defined in Flask
            await asyncio.sleep(1)  # Update every second
        self.reset_hour_block()

    def reset_hour_block(self):
        """Resets the hourly tracking block."""
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=1)

    # Notification methods
    def send_notification(self, message: str):
        """
        Sends notification through all configured channels: email, SMS, Telegram, and Discord.
        """
        contacts = self.db_handler.fetch("SELECT type, encrypted_value FROM communications")
        for contact in contacts:
            contact_type = contact["type"]
            contact_value = SecurityManager.decrypt_data(contact["encrypted_value"])

            if contact_type == "email":
                self.send_email(contact_value, message)
            elif contact_type == "sms":
                self.send_sms(contact_value, message)
            elif contact_type == "telegram":
                self.send_telegram(contact_value, message)
            elif contact_type == "discord":
                self.send_discord(contact_value, message)

    def send_email(self, recipient_email, message):
        """Sends an email notification."""
        sender_email = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASS")
        with smtplib.SMTP_SSL("smtp.example.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, recipient_email, f"Subject: NAV Alert\n\n{message}")

    def send_sms(self, recipient_phone, message):
        """Sends an SMS notification using Twilio."""
        if self.twilio_client:
            self.twilio_client.messages.create(
                body=message,
                from_=os.getenv("TWILIO_PHONE"),
                to=recipient_phone
            )

    def send_telegram(self, bot_token, message):
        """Sends a message via Telegram bot."""
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": os.getenv("TELEGRAM_CHAT_ID"), "text": message}
        requests.post(url, json=payload)

    def send_discord(self, webhook_url, message):
        """Sends a message to Discord webhook."""
        payload = {"content": message}
        requests.post(webhook_url, json=payload)
