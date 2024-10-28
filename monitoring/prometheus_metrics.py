from prometheus_client import Gauge, start_http_server
import psutil
import time
import logging

class PrometheusMetricsExporter:
    """Exporter for system and agent performance metrics using Prometheus."""

    def __init__(self, port=8000):
        """Initialize and start the Prometheus HTTP server."""
        start_http_server(port)
        self.cpu_usage_gauge = Gauge('cpu_usage', 'CPU usage percentage')
        self.memory_usage_gauge = Gauge('memory_usage', 'Memory usage percentage')
        self.disk_usage_gauge = Gauge('disk_usage', 'Disk usage percentage')
        self.network_usage_gauge = Gauge('network_usage', 'Network usage bytes sent/received')

    def collect_system_metrics(self):
        """Continuously collect system resource metrics."""
        while True:
            self.cpu_usage_gauge.set(psutil.cpu_percent(interval=1))
            self.memory_usage_gauge.set(psutil.virtual_memory().percent)
            self.disk_usage_gauge.set(psutil.disk_usage('/').percent)
            net_io = psutil.net_io_counters()
            self.network_usage_gauge.set(net_io.bytes_sent + net_io.bytes_recv)

            logging.info("Collected system metrics: CPU, Memory, Disk, Network.")
            time.sleep(5)  # Update metrics every 5 seconds
