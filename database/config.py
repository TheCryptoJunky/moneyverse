# Full file path: moneyverse/database/config.py

import os

DB_CONFIG = {
    'USER': os.getenv('DB_USER', 'username'),
    'PASSWORD': os.getenv('DB_PASSWORD', 'password'),
    'DATABASE': os.getenv('DB_DATABASE', 'moneyverse_db'),
    'HOST': os.getenv('DB_HOST', 'localhost'),
    'PORT': os.getenv('DB_PORT', 5432),
}

# Encryption key for sensitive data
ENCRYPTION_KEY = os.getenv('DB_ENCRYPTION_KEY', 'default_encryption_key')
