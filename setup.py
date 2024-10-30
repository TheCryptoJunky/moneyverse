from setuptools import setup, find_packages
import os
import subprocess

def run_mysql_setup():
    """Run MySQL setup script if necessary."""
    try:
        subprocess.run(["python3", "src/database/mysql_setup.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error setting up MySQL: {e}")
    except FileNotFoundError:
        print("MySQL setup script not found. Please ensure src/database/mysql_setup.py exists.")

# Execute MySQL setup on installation
run_mysql_setup()

setup(
    name="moneyverse_trading_framework",
    version="1.0",
    description="Trading bot framework with AI and reinforcement learning for MEV strategies",
    author="TheCryptoJunky",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "mysql-connector-python",
        "tensorflow>=2.0",
        "pandas",
        "ccxt",  # For exchange integration
        "python-dotenv",  # For environment variable management
        "flask",  # Flask GUI for bot management
        "gym",  # Reinforcement learning environments
        "numpy",
        # Include other dependencies as necessary
    ],
    scripts=["src/database/mysql_setup.py"],  # Ensures the setup script is included
    include_package_data=True,
)
