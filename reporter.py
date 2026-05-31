"""
reporter.py -- Sends risk-factor alerts from the Raspberry Pi to the ISERF web app.

Design goals:
  * Never block the detection loop on the network (each POST runs in its own
    daemon thread, with a short timeout and swallowed exceptions).
  * Edge-triggered: send one alert when a condition STARTS, and (optionally)
    one when it CLEARS -- not once per video frame.

Configuration is read from environment variables so no secrets live in code:
  ISERF_API_URL    e.g. https://your-app.vercel.app/api/alerts
  ISERF_API_KEY    must match DEVICE_API_KEY in the web app
  ISERF_DEVICE_ID  identifies this Pi/vehicle (default: 'pi-01')
  ISERF_DRIVER_ID  optional driver identifier

Usage (see integrated_test.py for wiring):

    from reporter import AlertReporter
    reporter = AlertReporter()
    reporter.update("drowsiness", active=eyes_closed, ear=ear_value)
    reporter.update("bpm_abnormal", active=bpm_abnormal, bpm=bpm)
"""

import os
import threading

try:
    import requests
except ImportError:  # keep the detection script runnable even without requests
    requests = None


class AlertReporter:
    def __init__(self, api_url=None, api_key=None, device_id=None, driver_id=None):
        self.api_url = api_url or os.environ.get("ISERF_API_URL", "")
        self.api_key = api_key or os.environ.get("ISERF_API_KEY", "")
        self.device_id = device_id or os.environ.get("ISERF_DEVICE_ID", "pi-01")
        self.driver_id = driver_id or os.environ.get("ISERF_DRIVER_ID")
        # Remembers whether each alert type is currently active (for edge detection).
        self._active = {}

        if not self.api_url or not self.api_key:
            print("[reporter] ISERF_API_URL / ISERF_API_KEY not set; "
                  "alerts will not be sent.")
        if requests is None:
            print("[reporter] 'requests' not installed; alerts will not be sent.")

    # -- public API -----------------------------------------------------------
    def update(self, alert_type, active, ear=None, bpm=None,
               severity="critical", message=None, report_clear=True):
        """
        Call every loop iteration with the current boolean state of a condition.
        Fires a network alert only on the rising edge (and falling edge if
        report_clear=True), not on every call.
        """
        was_active = self._active.get(alert_type, False)

        if active and not was_active:
            self._active[alert_type] = True
            self._send(alert_type, "active", severity, ear, bpm,
                       message or self._default_msg(alert_type, ear, bpm))
        elif not active and was_active:
            self._active[alert_type] = False
            if report_clear:
                self._send(alert_type, "cleared", "warning", ear, bpm,
                           f"{alert_type} cleared")

    # -- internals ------------------------------------------------------------
    def _default_msg(self, alert_type, ear, bpm):
        if alert_type == "drowsiness":
            return "Eyes closed beyond threshold" + (
                f" (EAR {ear:.3f})" if ear is not None else "")
        if alert_type == "bpm_abnormal":
            return "Heart rate out of normal range" + (
                f" ({bpm:.0f} bpm)" if bpm is not None else "")
        return alert_type

    def _send(self, alert_type, status, severity, ear, bpm, message):
        if requests is None or not self.api_url or not self.api_key:
            return

        payload = {
            "device_id": self.device_id,
            "driver_id": self.driver_id,
            "type": alert_type,
            "severity": severity,
            "status": status,
            "ear_value": float(ear) if ear is not None else None,
            "bpm_value": float(bpm) if bpm is not None else None,
            "message": message,
        }

        def _post():
            try:
                requests.post(
                    self.api_url,
                    json=payload,
                    headers={"x-api-key": self.api_key},
                    timeout=3,
                )
            except Exception as exc:  # never crash detection on a network error
                print(f"[reporter] failed to send {alert_type}/{status}: {exc}")

        threading.Thread(target=_post, daemon=True).start()
