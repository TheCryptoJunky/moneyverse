import os
import hashlib
from cryptography.fernet import Fernet
import pyotp
import qrcode
from centralized_logger import CentralizedLogger

# Initialize centralized logger for tracking security events
logger = CentralizedLogger()

# Initialize encryption key and cipher from environment or generate securely if absent
encryption_key = os.getenv('DB_ENCRYPTION_KEY')
if not encryption_key:
    encryption_key = Fernet.generate_key()  # Secure generation for production
    os.environ['DB_ENCRYPTION_KEY'] = encryption_key.decode()  # Store in environment for continuity
cipher = Fernet(encryption_key)

class SecurityManager:
    """
    Manages encryption, secure authentication, and password hashing for database operations.
    """

    @staticmethod
    def encrypt_data(data: str) -> str:
        """Encrypts sensitive data before storing securely in the database."""
        encrypted_data = cipher.encrypt(data.encode())
        return encrypted_data.decode()

    @staticmethod
    def decrypt_data(data: str) -> str:
        """Decrypts sensitive data retrieved from the database."""
        decrypted_data = cipher.decrypt(data.encode())
        return decrypted_data.decode()

    @staticmethod
    def hash_password(password: str) -> str:
        """Hashes passwords securely for storage using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()

class UserAuthManager:
    """
    Manages TOTP-based two-factor authentication (2FA) and generates QR codes for user setup.
    """

    def __init__(self):
        self.totp = None

    def setup_2fa(self, user_identifier: str):
        """
        Sets up TOTP-based 2FA for a user, generating a QR code for scanning into an authenticator app.
        
        Parameters:
            user_identifier (str): The unique identifier for the user setting up 2FA.
        """
        secret = pyotp.random_base32()  # Generate a random base32 secret for TOTP
        self.totp = pyotp.TOTP(secret)
        qr_url = self.totp.provisioning_uri(user_identifier, issuer_name="Moneyverse Security")

        # Generate and display QR code for easy setup in authenticator app
        qr = qrcode.make(qr_url)
        qr.show()  # Show the QR code to user for scanning
        logger.log("info", "2FA setup complete. Scan the QR code in an authenticator app.")

    def verify_2fa(self, token: str) -> bool:
        """
        Verifies a TOTP-based 2FA token to authenticate user actions.

        Parameters:
            token (str): The token provided by the user from their authenticator app.

        Returns:
            bool: True if the token is valid; False otherwise.
        """
        if self.totp and self.totp.verify(token):
            logger.log("info", "2FA verification successful.")
            return True
        else:
            logger.log("warning", "Invalid 2FA token. Verification failed.")
            return False
