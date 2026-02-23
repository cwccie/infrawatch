"""Notification backends for alert delivery.

Implements webhook, email, Slack, and PagerDuty notification channels.
All notifiers implement the Notifier protocol.
"""

from __future__ import annotations

import json
import logging
import smtplib
from email.mime.text import MIMEText
from typing import Optional

logger = logging.getLogger(__name__)


class WebhookNotifier:
    """Sends alerts via HTTP webhook (POST with JSON body).

    Args:
        url: Webhook endpoint URL.
        headers: Additional HTTP headers.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 10.0,
    ):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout

    def send(self, alert) -> bool:
        import requests

        try:
            resp = requests.post(
                self.url,
                json=alert.to_dict(),
                headers=self.headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.error("Webhook notification failed: %s", exc)
            return False


class EmailNotifier:
    """Sends alerts via email (SMTP).

    Args:
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port.
        from_addr: Sender email address.
        to_addrs: Recipient email addresses.
        username: SMTP authentication username.
        password: SMTP authentication password.
        use_tls: Whether to use STARTTLS.
    """

    def __init__(
        self,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        from_addr: str = "",
        to_addrs: Optional[list[str]] = None,
        username: str = "",
        password: str = "",
        use_tls: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.from_addr = from_addr
        self.to_addrs = to_addrs or []
        self.username = username
        self.password = password
        self.use_tls = use_tls

    def send(self, alert) -> bool:
        try:
            subject = f"[InfraWatch {alert.severity.label.upper()}] {alert.metric_name}"
            body = (
                f"Alert: {alert.message}\n\n"
                f"Metric: {alert.metric_name}\n"
                f"Value: {alert.value:.4f}\n"
                f"Score: {alert.score:.4f}\n"
                f"Severity: {alert.severity.label}\n"
                f"State: {alert.state.value}\n"
                f"Labels: {alert.labels}\n"
                f"Duration: {alert.duration_minutes:.1f} minutes\n"
                f"Count: {alert.count}\n"
            )

            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username:
                    server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            return True
        except Exception as exc:
            logger.error("Email notification failed: %s", exc)
            return False


class SlackNotifier:
    """Sends alerts to Slack via incoming webhook.

    Args:
        webhook_url: Slack incoming webhook URL.
        channel: Optional channel override.
        username: Bot username.
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str = "",
        username: str = "InfraWatch",
    ):
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username

    def send(self, alert) -> bool:
        import requests

        severity_colors = {
            "info": "#36a64f",
            "low": "#daa520",
            "medium": "#ff8c00",
            "high": "#ff4500",
            "critical": "#dc143c",
        }

        color = severity_colors.get(alert.severity.label, "#808080")

        payload = {
            "username": self.username,
            "attachments": [
                {
                    "color": color,
                    "title": f"{alert.severity.label.upper()}: {alert.metric_name}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Value", "value": f"{alert.value:.4f}", "short": True},
                        {"title": "Score", "value": f"{alert.score:.4f}", "short": True},
                        {"title": "Duration", "value": f"{alert.duration_minutes:.1f}m", "short": True},
                        {"title": "Count", "value": str(alert.count), "short": True},
                    ],
                    "footer": "InfraWatch Anomaly Detection",
                }
            ],
        }

        if self.channel:
            payload["channel"] = self.channel

        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            return resp.status_code == 200
        except Exception as exc:
            logger.error("Slack notification failed: %s", exc)
            return False


class PagerDutyNotifier:
    """Sends alerts to PagerDuty via Events API v2.

    Args:
        routing_key: PagerDuty integration routing key.
        api_url: PagerDuty Events API endpoint.
    """

    def __init__(
        self,
        routing_key: str,
        api_url: str = "https://events.pagerduty.com/v2/enqueue",
    ):
        self.routing_key = routing_key
        self.api_url = api_url

    def send(self, alert) -> bool:
        import requests

        severity_map = {
            "info": "info",
            "low": "warning",
            "medium": "warning",
            "high": "error",
            "critical": "critical",
        }

        payload = {
            "routing_key": self.routing_key,
            "event_action": "trigger" if alert.state.value == "firing" else "resolve",
            "dedup_key": alert.fingerprint,
            "payload": {
                "summary": alert.message,
                "source": alert.metric_name,
                "severity": severity_map.get(alert.severity.label, "warning"),
                "custom_details": {
                    "value": alert.value,
                    "score": alert.score,
                    "labels": alert.labels,
                    "count": alert.count,
                },
            },
        }

        try:
            resp = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            return resp.status_code == 202
        except Exception as exc:
            logger.error("PagerDuty notification failed: %s", exc)
            return False
