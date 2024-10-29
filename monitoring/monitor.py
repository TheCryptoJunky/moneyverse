# Full file path: /moneyverse/monitoring/monitor.py

import os
import smtplib
import asyncio
from email.mime.text import MIMEText
from ai.agents.swarm_protection_manager import SwarmProtectionManager
from ai.models.rl_nav_optimizer import RLNavOptimizer
from centralized_logger import CentralizedLogger
from database_manager import fetch_performance_data

# Initialize classes and constants
logger = CentralizedLogger()
swarm_protection_manager = SwarmProtectionManager()
nav_optimizer = RLNavOptimizer()
alert_email = os.getenv("ALERT_EMAIL")

async def take_corrective_action(metric_name, value):
    """
    Executes corrective actions using AI-driven managers and optimizers.
    """
    if metric_name == "NAV Growth":
        corrective_action = await nav_optimizer.optimize_strategy(current_nav=value)
        logger.log_warning(f"Corrective action for NAV initiated: {corrective_action}")
    elif metric_name == "Attack Detected":
        protection_strategy = await swarm_protection_manager.select_protection_strategy()
        logger.log_critical(f"Counter-protection strategy activated: {protection_strategy}")
        # Trigger real-time MEV counter-protection strategy

async def monitor_swarm_performance():
    """
    Monitors swarm performance by checking metrics and triggering alerts or corrective actions if needed.
    """
    performance_data = await fetch_performance_data()
    for metric in performance_data:
        metric_name, value = metric["metric_name"], metric["value"]

        if metric_name == "NAV Growth" and value < target_nav_growth:
            logger.log_warning(f"NAV growth below target: {value}%")
            await take_corrective_action(metric_name, value)
            await send_alert("Low NAV Growth Alert", f"NAV growth below target: {value}%")
        
        elif metric_name == "Attack Detected":
            logger.log_critical(f"Attack detected: {metric.get('description', '')}")
            await take_corrective_action(metric_name, value)
            await send_alert("Attack Alert", f"Attack detected: {metric.get('description', '')}")

async def send_alert(subject, message):
    """ Sends an email alert for critical events. """
    if not alert_email:
        logger.log_warning("Alert email not configured.")
        return

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = os.getenv("SMTP_USER")
    msg["To"] = alert_email

    try:
        with smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT", 587))) as server:
            server.starttls()
            server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"))
            server.sendmail(os.getenv("SMTP_USER"), alert_email, msg.as_string())
            logger.log_info(f"Alert sent to {alert_email}: {subject}")
    except Exception as e:
        logger.log_error(f"Failed to send alert email: {e}")

if __name__ == "__main__":
    asyncio.run(monitor_swarm_performance())
