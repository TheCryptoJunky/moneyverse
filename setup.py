# File: setup.py
from setuptools import setup
import os

# Run MySQL setup after installation
os.system("python3 /src/database/mysql_setup.py")

setup(
    name="trading_framework",
    version="1.0",
    description="Trading bot framework with AI and reinforcement learning",
    author="Your Name",
    packages=["src", "src.ai", "src.database", "src.managers", "src.trading"],
    install_requires=[
        "mysql-connector-python",
        "tensorflow",
        "pandas",
        "ccxt",
        "dotenv",
        # other dependencies...
    ],
    scripts=["/src/database/mysql_setup.py"]  # Run database setup as part of installation
)
