"""REST API (Flask).

Query metrics, anomaly status, forecasts, silence alerts, and manage
maintenance windows.
"""

from infrawatch.api.app import create_app

__all__ = ["create_app"]
