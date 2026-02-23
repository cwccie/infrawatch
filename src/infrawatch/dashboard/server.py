"""Dashboard server with embedded HTML/JS visualization.

Serves a single-page dashboard that renders metric graphs, anomaly
overlays, forecast projections, and maintenance windows using Chart.js
via CDN (no build step required).
"""

from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np
from flask import Flask, render_template_string

from infrawatch.api.app import create_app


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>InfraWatch Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #0d1117; color: #c9d1d9; }
        .header { background: #161b22; padding: 16px 24px; border-bottom: 1px solid #30363d;
                  display: flex; align-items: center; justify-content: space-between; }
        .header h1 { font-size: 20px; color: #58a6ff; }
        .header .status { display: flex; gap: 16px; }
        .status-badge { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; }
        .status-healthy { background: #238636; color: #fff; }
        .status-alerting { background: #da3633; color: #fff; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 16px; padding: 24px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 16px; }
        .card h2 { font-size: 14px; color: #8b949e; margin-bottom: 12px; text-transform: uppercase;
                   letter-spacing: 0.5px; }
        .metric-value { font-size: 36px; font-weight: 700; color: #58a6ff; }
        .chart-container { position: relative; height: 250px; }
        .alert-list { list-style: none; }
        .alert-item { padding: 8px 12px; border-left: 3px solid; margin-bottom: 8px;
                     background: #0d1117; border-radius: 0 4px 4px 0; }
        .alert-critical { border-color: #da3633; }
        .alert-high { border-color: #f85149; }
        .alert-medium { border-color: #d29922; }
        .alert-low { border-color: #58a6ff; }
        .maint-window { padding: 8px; background: #1c2333; border-radius: 4px; margin-bottom: 8px;
                        border-left: 3px solid #3fb950; }
        .footer { text-align: center; padding: 16px; color: #484f58; font-size: 12px; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; padding: 12px; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>InfraWatch</h1>
        <div class="status">
            <span class="status-badge status-healthy" id="health-badge">Healthy</span>
            <span style="color: #8b949e; font-size: 13px;" id="update-time"></span>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>System Overview</h2>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
                <div>
                    <div style="color: #8b949e; font-size: 12px;">Metrics</div>
                    <div class="metric-value" id="metric-count">-</div>
                </div>
                <div>
                    <div style="color: #8b949e; font-size: 12px;">Active Alerts</div>
                    <div class="metric-value" id="alert-count" style="color: #3fb950;">-</div>
                </div>
                <div>
                    <div style="color: #8b949e; font-size: 12px;">Maintenance</div>
                    <div class="metric-value" id="maint-count" style="color: #d29922;">-</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Metric Trend</h2>
            <div class="chart-container">
                <canvas id="metricChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Active Alerts</h2>
            <ul class="alert-list" id="alert-list">
                <li style="color: #484f58;">No active alerts</li>
            </ul>
        </div>

        <div class="card">
            <h2>Maintenance Windows</h2>
            <div id="maint-list">
                <div style="color: #484f58;">No scheduled maintenance</div>
            </div>
        </div>
    </div>

    <div class="footer">
        InfraWatch v0.1.0 &mdash; Anomaly Detection for Infrastructure Metrics
    </div>

    <script>
        const ctx = document.getElementById('metricChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Metric Value',
                    data: [],
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                }, {
                    label: 'Anomaly',
                    data: [],
                    borderColor: '#da3633',
                    backgroundColor: '#da3633',
                    pointRadius: 6,
                    pointStyle: 'triangle',
                    showLine: false,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: '#484f58' }, grid: { color: '#21262d' } },
                    y: { ticks: { color: '#484f58' }, grid: { color: '#21262d' } }
                },
                plugins: { legend: { labels: { color: '#c9d1d9' } } }
            }
        });

        async function refresh() {
            try {
                const health = await fetch('/health').then(r => r.json());
                document.getElementById('update-time').textContent =
                    'Updated: ' + new Date().toLocaleTimeString();

                const metrics = await fetch('/api/v1/metrics').then(r => r.json());
                document.getElementById('metric-count').textContent = metrics.metrics.length;

                const alertsResp = await fetch('/api/v1/alerts').then(r => r.json());
                const alertCount = alertsResp.count;
                document.getElementById('alert-count').textContent = alertCount;
                document.getElementById('alert-count').style.color =
                    alertCount > 0 ? '#da3633' : '#3fb950';

                const badge = document.getElementById('health-badge');
                if (alertCount > 0) {
                    badge.className = 'status-badge status-alerting';
                    badge.textContent = alertCount + ' Alert(s)';
                } else {
                    badge.className = 'status-badge status-healthy';
                    badge.textContent = 'Healthy';
                }

                // Update alert list
                const alertList = document.getElementById('alert-list');
                if (alertsResp.alerts.length > 0) {
                    alertList.innerHTML = alertsResp.alerts.map(a =>
                        `<li class="alert-item alert-${a.severity}">
                            <strong>${a.severity.toUpperCase()}</strong>: ${a.metric_name}
                            <br><small>Score: ${a.score.toFixed(3)} | Value: ${a.value.toFixed(2)}</small>
                        </li>`
                    ).join('');
                }

                // Load first metric chart data
                if (metrics.metrics.length > 0) {
                    const name = metrics.metrics[0].name;
                    const mdata = await fetch('/api/v1/metrics/' + name).then(r => r.json());
                    const points = mdata.points.slice(-200);
                    chart.data.labels = points.map((p, i) => i);
                    chart.data.datasets[0].data = points.map(p => p.value);
                    chart.update('none');
                }

                const maint = await fetch('/api/v1/maintenance').then(r => r.json());
                document.getElementById('maint-count').textContent = maint.windows.length;
                const maintList = document.getElementById('maint-list');
                if (maint.windows.length > 0) {
                    maintList.innerHTML = maint.windows.map(w =>
                        `<div class="maint-window">
                            <strong>${w.name}</strong>
                            <br><small>${w.duration_minutes.toFixed(0)} min |
                            ${w.is_active ? 'ACTIVE' : 'Scheduled'}</small>
                        </div>`
                    ).join('');
                }
            } catch (e) {
                console.error('Dashboard refresh error:', e);
            }
        }

        refresh();
        setInterval(refresh, 10000);
    </script>
</body>
</html>
"""


def create_dashboard_app(
    detection_pipeline=None,
    alert_engine=None,
    forecast_engine=None,
    maintenance_manager=None,
    port: int = 8080,
) -> Flask:
    """Create a Flask app that serves both API and dashboard.

    Args:
        detection_pipeline: Detection pipeline instance.
        alert_engine: Alert engine instance.
        forecast_engine: Forecast engine instance.
        maintenance_manager: Maintenance manager instance.
        port: Port to bind to (informational — actual binding done by caller).

    Returns:
        Flask app with API routes and dashboard.
    """
    app = create_app(
        detection_pipeline=detection_pipeline,
        alert_engine=alert_engine,
        forecast_engine=forecast_engine,
        maintenance_manager=maintenance_manager,
    )

    @app.route("/")
    def dashboard():
        return render_template_string(DASHBOARD_HTML)

    return app
