from prometheus_client import Counter, Summary, Gauge
import logging

class AgentPerformanceExporter:
    """Exporter for agent-specific performance metrics using Prometheus."""

    def __init__(self):
        """Initialize Prometheus metrics for agent performance."""
        self.total_trades = Counter('total_trades', 'Total number of trades executed by the agent')
        self.successful_trades = Counter('successful_trades', 'Number of successful trades')
        self.failed_trades = Counter('failed_trades', 'Number of failed trades')
        self.profit_loss = Gauge('total_profit_loss', 'Total profit or loss')
        self.execution_time = Summary('execution_time', 'Time spent executing agent trades')

    def update_metrics(self, metrics):
        """Update Prometheus metrics based on the latest agent performance data."""
        self.total_trades.inc(metrics['total_trades'])
        self.successful_trades.inc(metrics['successful_trades'])
        self.failed_trades.inc(metrics['failed_trades'])
        self.profit_loss.set(metrics['total_profit_loss'])
        self.execution_time.observe(metrics['execution_time'])

        logging.info(f"Exported agent metrics: {metrics}")
