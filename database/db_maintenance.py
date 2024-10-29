import os
import subprocess
from datetime import datetime
from database.config import DB_CONFIG, BACKUP_PATH
from centralized_logger import CentralizedLogger
import psutil  # Optional: For detailed performance monitoring

logger = CentralizedLogger()

class DBMaintenance:
    """
    Manages automated database tasks, including backup, indexing, and performance monitoring.
    """

    def __init__(self):
        self.backup_path = BACKUP_PATH
        self.db_name = DB_CONFIG['DATABASE']

    def backup_database(self):
        """
        Creates an automated database backup, saved with a timestamp in the specified backup path.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_file = os.path.join(self.backup_path, f"backup_{timestamp}.sql")
        
        try:
            # Run the backup command
            subprocess.run(
                ["pg_dump", "-U", DB_CONFIG['USER'], self.db_name, "-f", backup_file],
                check=True,
                env={"PGPASSWORD": DB_CONFIG['PASSWORD']}  # Secure password handling
            )
            logger.log("info", f"Database backup created successfully at {backup_file}")
        except subprocess.CalledProcessError as e:
            logger.log("error", f"Database backup failed: {e}")

    def optimize_tables(self):
        """
        Re-indexes tables to optimize database performance.
        """
        try:
            command = f'psql -U {DB_CONFIG["USER"]} -d {self.db_name} -c "REINDEX DATABASE {self.db_name};"'
            subprocess.run(command, shell=True, check=True, env={"PGPASSWORD": DB_CONFIG['PASSWORD']})
            logger.log("info", "Database indexing completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.log("error", f"Database indexing failed: {e}")

    def monitor_performance(self):
        """
        Tracks and logs database performance metrics, including CPU and memory usage.
        """
        try:
            # Log database-specific resource usage (optional detailed monitoring)
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            logger.log("info", f"Database performance monitored. CPU: {cpu_usage}%, Memory: {memory_usage}%")
        except Exception as e:
            logger.log("error", f"Performance monitoring failed: {e}")

    def scheduled_maintenance(self):
        """
        Runs scheduled maintenance tasks: backup, optimize, and monitor.
        Can be run periodically by a scheduler.
        """
        logger.log("info", "Starting scheduled database maintenance.")
        self.backup_database()
        self.optimize_tables()
        self.monitor_performance()
        logger.log("info", "Scheduled maintenance completed successfully.")

