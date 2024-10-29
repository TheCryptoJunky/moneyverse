# Full file path: /moneyverse/tests/test_protection.py

import unittest
from unittest.mock import patch
from strategies.protection import activate_partner_protection

class TestPartnerProtection(unittest.TestCase):
    @patch("strategies.protection.perform_counter_sandwich")
    def test_activate_counter_sandwich(self, mock_counter_sandwich):
        """
        Test counter-sandwich strategy activation.
        """
        mock_counter_sandwich.return_value = True  # Mock successful execution
        
        result = activate_partner_protection("0xPartnerToken", ["counter_sandwich"])
        self.assertTrue(result, "Counter-sandwich strategy failed to activate.")
        
        # Assert that the counter-sandwich strategy was called once
        mock_counter_sandwich.assert_called_once_with("0xPartnerToken")

    @patch("strategies.protection.perform_counter_sandwich")
    @patch("strategies.protection.perform_front_run")
    def test_activate_multiple_attacks(self, mock_front_run, mock_counter_sandwich):
        """
        Test activation of multiple protection strategies.
        """
        mock_counter_sandwich.return_value = True
        mock_front_run.return_value = True
        
        result = activate_partner_protection("0xPartnerToken", ["counter_sandwich", "front_run"])
        self.assertTrue(result, "Failed to activate multiple protection strategies.")
        
        # Assert that both strategies were called
        mock_counter_sandwich.assert_called_once_with("0xPartnerToken")
        mock_front_run.assert_called_once_with("0xPartnerToken")

    @patch("strategies.protection.perform_back_run")
    def test_activate_back_run_failure(self, mock_back_run):
        """
        Test back-running strategy activation failure handling.
        """
        mock_back_run.side_effect = Exception("Simulated back-run failure")
        
        result = activate_partner_protection("0xPartnerToken", ["back_run"])
        self.assertFalse(result, "Back-run strategy should fail due to simulated error.")
        
        # Confirm that an exception in back-run strategy was handled
        mock_back_run.assert_called_once_with("0xPartnerToken")

if __name__ == "__main__":
    unittest.main()
