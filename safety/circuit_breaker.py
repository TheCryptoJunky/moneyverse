# File: /src/safety/circuit_breaker.py

from src.ai.ai_helpers import CircuitBreakerHelper  # Corrected import path

class CircuitBreaker:
    """
    A Circuit Breaker to halt trading during risky situations or when predefined thresholds are met.
    """

    def __init__(self, risk_threshold=0.8):
        """
        Initialize the Circuit Breaker with a risk threshold.

        :param risk_threshold: The risk score threshold to trigger the breaker.
        """
        self.risk_threshold = risk_threshold

    def check_risk(self, risk_score):
        """
        Check if the current risk score exceeds the threshold, triggering the circuit breaker.

        :param risk_score: The calculated risk score.
        :return: True if the circuit breaker is triggered, False otherwise.
        """
        if risk_score > self.risk_threshold:
            self.trigger_circuit_breaker()
            return True
        return False

    def trigger_circuit_breaker(self):
        """
        Trigger the circuit breaker, halting all trades.
        """
        print("Circuit breaker triggered! Halting all trades.")
        # Here we would log this action and perform necessary trade halting logic.

    def reset_circuit_breaker(self):
        """
        Reset the circuit breaker once the risk is reduced.
        """
        print("Circuit breaker reset. Resuming normal operations.")
        # Log and reset actions to resume trading.
