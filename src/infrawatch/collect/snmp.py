"""SNMP metric poller.

Polls network devices via SNMP GET/WALK and converts OID values into
InfraWatch metrics. Handles counter wrapping for 32-bit and 64-bit counters.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from infrawatch.collect.metric import Metric, MetricBatch

logger = logging.getLogger(__name__)

# Common SNMP OID to friendly-name mappings
COMMON_OIDS = {
    "1.3.6.1.2.1.1.3.0": ("sysUpTime", "ticks"),
    "1.3.6.1.2.1.2.2.1.10": ("ifInOctets", "bytes"),
    "1.3.6.1.2.1.2.2.1.16": ("ifOutOctets", "bytes"),
    "1.3.6.1.2.1.2.2.1.14": ("ifInErrors", "errors"),
    "1.3.6.1.2.1.2.2.1.20": ("ifOutErrors", "errors"),
    "1.3.6.1.2.1.25.3.3.1.2": ("hrProcessorLoad", "percent"),
    "1.3.6.1.2.1.25.2.3.1.6": ("hrStorageUsed", "units"),
    "1.3.6.1.4.1.2021.11.9.0": ("ssCpuUser", "percent"),
    "1.3.6.1.4.1.2021.11.10.0": ("ssCpuSystem", "percent"),
    "1.3.6.1.4.1.2021.11.11.0": ("ssCpuIdle", "percent"),
    "1.3.6.1.4.1.2021.4.5.0": ("memTotalReal", "kB"),
    "1.3.6.1.4.1.2021.4.6.0": ("memAvailReal", "kB"),
}

COUNTER32_MAX = 2**32
COUNTER64_MAX = 2**64


@dataclass
class SNMPTarget:
    """Configuration for an SNMP-pollable device."""

    host: str
    port: int = 161
    community: str = "public"
    version: int = 2  # 1, 2, or 3
    oids: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)


def unwrap_counter(current: float, previous: float, bits: int = 64) -> float:
    """Compute the delta between two counter readings, handling wrap-around.

    Args:
        current: Current counter value.
        previous: Previous counter value.
        bits: Counter width (32 or 64).

    Returns:
        Non-negative delta accounting for counter wrap.
    """
    max_val = COUNTER64_MAX if bits == 64 else COUNTER32_MAX
    if current >= previous:
        return current - previous
    # Counter wrapped
    return (max_val - previous) + current


class SNMPPoller:
    """Polls SNMP targets and produces InfraWatch metrics.

    For production use, requires the pysnmp optional dependency.
    Includes a simulation mode for testing without actual SNMP infrastructure.

    Args:
        targets: List of SNMPTarget configurations.
        counter_bits: Default counter width for unwrapping (32 or 64).
    """

    def __init__(
        self,
        targets: Optional[list[SNMPTarget]] = None,
        counter_bits: int = 64,
    ):
        self.targets = targets or []
        self.counter_bits = counter_bits
        self._previous_values: dict[str, dict[str, float]] = {}

    def poll_simulated(
        self,
        target: SNMPTarget,
        oid_values: dict[str, float],
    ) -> MetricBatch:
        """Process pre-fetched OID values as if they came from SNMP.

        This is the primary interface for testing and for environments where
        SNMP data is provided through other means (e.g., SNMP trap receivers).

        Args:
            target: Target device configuration.
            oid_values: Mapping of OID string to numeric value.

        Returns:
            MetricBatch of converted metrics.
        """
        batch = MetricBatch(collector_id=f"snmp-{target.host}")
        now = time.time()
        host_key = f"{target.host}:{target.port}"

        prev = self._previous_values.get(host_key, {})

        for oid, value in oid_values.items():
            # Resolve friendly name
            base_oid = oid.rsplit(".", 1)[0] if "." in oid else oid
            name, unit = COMMON_OIDS.get(base_oid, (oid, ""))

            # If this is a counter OID and we have a previous value, compute rate
            metric_value = value
            if base_oid in prev and "Octets" in name:
                delta = unwrap_counter(value, prev[base_oid], self.counter_bits)
                metric_value = delta
                name = name + "_delta"

            prev[base_oid] = value

            labels = {**target.labels, "host": target.host}
            # Extract interface index from OID suffix
            if "." in oid:
                suffix = oid.rsplit(".", 1)[1]
                if suffix.isdigit():
                    labels["ifIndex"] = suffix

            batch.add(Metric(
                name=name,
                value=metric_value,
                timestamp=now,
                labels=labels,
                source=f"snmp://{target.host}:{target.port}",
                unit=unit,
            ))

        self._previous_values[host_key] = prev
        return batch

    def poll(self, target: Optional[SNMPTarget] = None) -> MetricBatch:
        """Poll a target via SNMP. Requires pysnmp.

        Args:
            target: Device to poll. If None, polls the first configured target.

        Returns:
            MetricBatch with polled metrics.
        """
        t = target or (self.targets[0] if self.targets else None)
        if t is None:
            raise ValueError("No SNMP target provided or configured")

        try:
            from pysnmp.hlapi import (
                SnmpEngine, CommunityData, UdpTransportTarget,
                ContextData, ObjectType, ObjectIdentity, getCmd,
            )
        except ImportError:
            raise ImportError(
                "pysnmp is required for live SNMP polling. "
                "Install with: pip install infrawatch[snmp]"
            )

        oid_values: dict[str, float] = {}
        engine = SnmpEngine()

        for oid in t.oids:
            error_indication, error_status, error_index, var_binds = next(
                getCmd(
                    engine,
                    CommunityData(t.community),
                    UdpTransportTarget((t.host, t.port)),
                    ContextData(),
                    ObjectType(ObjectIdentity(oid)),
                )
            )

            if error_indication:
                logger.error("SNMP error for %s on %s: %s", oid, t.host, error_indication)
                continue
            if error_status:
                logger.error(
                    "SNMP error status for %s on %s: %s at %s",
                    oid, t.host, error_status.prettyPrint(),
                    var_binds[int(error_index) - 1][0] if error_index else "?",
                )
                continue

            for var_bind in var_binds:
                oid_str = str(var_bind[0])
                try:
                    oid_values[oid_str] = float(var_bind[1])
                except (ValueError, TypeError):
                    logger.warning("Non-numeric SNMP value for %s: %s", oid_str, var_bind[1])

        return self.poll_simulated(t, oid_values)

    def poll_all(self) -> MetricBatch:
        """Poll all configured targets."""
        combined = MetricBatch(collector_id="snmp-multi")
        for target in self.targets:
            try:
                batch = self.poll(target)
                for m in batch:
                    combined.add(m)
            except Exception as exc:
                logger.error("Failed to poll %s: %s", target.host, exc)
        return combined
