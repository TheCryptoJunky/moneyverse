# Full file path: /moneyverse/tests/test_flask_gui.py

import unittest
from flask import Flask
from flask_testing import TestCase
from flask_gui.dashboard import app

class TestFlaskGUI(TestCase):
    def create_app(self):
        app.config['TESTING'] = True
        return app

    # --- Tests for Pumplist Management ---
    
    def test_add_pumplist_entry(self):
        response = self.client.post('/pumplist/add', json={
            "entry": "0xTestToken",
            "duration": 60,
            "strategies": ["market_maker"]
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Entry added to Pumplist", response.data)

    def test_remove_pumplist_entry(self):
        response = self.client.post('/pumplist/remove', json={"entry": "0xTestToken"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Entry removed from Pumplist", response.data)

    # --- Tests for Redlist Management ---
    
    def test_add_redlist_entry(self):
        response = self.client.post('/redlist/add', json={"bad_actor": "0xMaliciousActor"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Bad actor added to Redlist", response.data)

    # --- Tests for Performance Analytics ---
    
    def test_performance_analytics(self):
        response = self.client.get('/performance/analytics')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Performance analytics retrieved successfully", response.data)

    # --- Tests for Framework Control (Start/Stop) ---
    
    def test_framework_control_start(self):
        response = self.client.post('/framework/control', json={"action": "start"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Framework started", response.data)

    def test_framework_control_stop(self):
        response = self.client.post('/framework/control', json={"action": "stop"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Framework stopped", response.data)

    def test_framework_control_invalid_action(self):
        response = self.client.post('/framework/control', json={"action": "invalid"})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Invalid action specified", response.data)

    # --- Tests for Configuration Update ---
    
    def test_update_configuration(self):
        response = self.client.post('/config/update', json={
            "nav_target": 2.0,
            "dca_interval": 300
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Configuration updated", response.data)
    
    # --- Tests for Strategy Refinement ---
    
    def test_refine_strategies(self):
        response = self.client.post('/strategies/refine')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Strategies refined", response.data)

    # --- Tests for Sensitive Dashboard Access Control ---
    
    def test_sensitive_dashboard_access_without_auth(self):
        response = self.client.get('/sensitive_dashboard')
        self.assertEqual(response.status_code, 403)  # Unauthorized without 2FA

if __name__ == "__main__":
    unittest.main()
