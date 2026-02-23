"""Tests for maintenance window management."""

import time
from infrawatch.maintenance.manager import MaintenanceWindow, MaintenanceManager


class TestMaintenanceWindow:
    def test_create_window(self):
        now = time.time()
        window = MaintenanceWindow(
            id="maint-1",
            name="Router upgrade",
            start_time=now - 60,
            end_time=now + 3600,
            targets=["router-01"],
        )
        assert window.is_active
        assert not window.is_past
        assert window.duration_minutes == pytest.approx(61.0, abs=0.1)

    def test_future_window(self):
        future = time.time() + 86400
        window = MaintenanceWindow(
            id="maint-2",
            name="Future maintenance",
            start_time=future,
            end_time=future + 3600,
        )
        assert window.is_future
        assert not window.is_active

    def test_matches_metric_global(self):
        now = time.time()
        window = MaintenanceWindow(
            id="maint-3", name="Global", start_time=now, end_time=now + 3600,
        )
        assert window.matches_metric("any_metric")

    def test_matches_metric_targeted(self):
        now = time.time()
        window = MaintenanceWindow(
            id="maint-4", name="Router", start_time=now, end_time=now + 3600,
            targets=["router"],
        )
        assert window.matches_metric("router_cpu")
        assert not window.matches_metric("server_cpu")

    def test_matches_label(self):
        now = time.time()
        window = MaintenanceWindow(
            id="maint-5", name="Host", start_time=now, end_time=now + 3600,
            targets=["web-01"],
        )
        assert window.matches_metric("cpu", {"host": "web-01"})
        assert not window.matches_metric("cpu", {"host": "web-02"})


class TestMaintenanceManager:
    def test_add_and_list(self):
        mgr = MaintenanceManager()
        now = time.time()
        mgr.add_window(MaintenanceWindow(
            id="m1", name="Test", start_time=now, end_time=now + 3600,
        ))
        windows = mgr.list_windows()
        assert len(windows) == 1

    def test_is_suppressed(self):
        mgr = MaintenanceManager()
        now = time.time()
        mgr.add_window(MaintenanceWindow(
            id="m1", name="Test", start_time=now - 60, end_time=now + 3600,
            targets=["cpu"],
        ))
        assert mgr.is_suppressed("cpu_usage")
        assert not mgr.is_suppressed("disk_usage")

    def test_remove_window(self):
        mgr = MaintenanceManager()
        now = time.time()
        mgr.add_window(MaintenanceWindow(
            id="m1", name="Test", start_time=now, end_time=now + 3600,
        ))
        assert mgr.remove_window("m1")
        assert len(mgr.list_windows()) == 0

    def test_active_windows(self):
        mgr = MaintenanceManager()
        now = time.time()
        mgr.add_window(MaintenanceWindow(
            id="m1", name="Active", start_time=now - 60, end_time=now + 3600,
        ))
        mgr.add_window(MaintenanceWindow(
            id="m2", name="Future", start_time=now + 7200, end_time=now + 10800,
        ))
        active = mgr.active_windows()
        assert len(active) == 1
        assert active[0].id == "m1"


import pytest
