"""
AUDI Type A — GPIO Alert System

Controls Raspberry Pi GPIO pins for:
  - ALERT:   Buzzer/relay on configured alert levels
  - STROBE:  Blinking LED during an active alert
  - RESET:   Physical button to clear the alert
  - REC_LED: Recording indicator (on while capturing)
  - REC_BTN: Toggle recording on/off
  - PAUSE_BTN: Pause recording for 5 minutes
  - FIELD_TAG: Green/yellow/red operator labels for field detections

Auto-detects Pi hardware vs dev machine (mock fallback).
"""

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable

logger = logging.getLogger("audi.gpio")

# ---------------------------------------------------------------------------


def _optional_pin(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


class GPIOController:
    """Manages GPIO pins for alerts, recording, and input buttons."""

    def __init__(self, config: dict):
        gpio_cfg = config.get("gpio", {})
        self.enabled = gpio_cfg.get("enabled", True)

        # Alarm pins
        self.alert_pin = _optional_pin(gpio_cfg.get("alert_pin", 2))
        self.strobe_pin = _optional_pin(gpio_cfg.get("strobe_pin", 24))
        self.reset_pin = _optional_pin(gpio_cfg.get("reset_pin", 23))

        # Recording pins
        self.record_led_pin = _optional_pin(gpio_cfg.get("record_led_pin"))
        self.record_button_pin = _optional_pin(gpio_cfg.get("record_button_pin"))
        self.pause_button_pin = _optional_pin(gpio_cfg.get("pause_button_pin", 18))
        self.field_tag_button_pins = {
            "green": _optional_pin(gpio_cfg.get("field_tag_green_pin", 22)),
            "yellow": _optional_pin(gpio_cfg.get("field_tag_yellow_pin", 27)),
            "red": _optional_pin(gpio_cfg.get("field_tag_red_pin", 17)),
        }

        self.alert_duration_ms = gpio_cfg.get("alert_duration_ms", 5000)
        self.pulse_interval_ms = gpio_cfg.get("pulse_interval_ms", 500)
        self.buzzer_patterns = {
            "RED_ALERT": {
                "on_ms": gpio_cfg.get("red_buzzer_on_ms", 120),
                "off_ms": gpio_cfg.get("red_buzzer_off_ms", 80),
            },
            "BLUE_ALERT": {
                "on_ms": gpio_cfg.get("blue_buzzer_on_ms", 420),
                "off_ms": gpio_cfg.get("blue_buzzer_off_ms", 280),
            },
            "UNKNOWN_ALERT": {
                "on_ms": gpio_cfg.get("unknown_buzzer_on_ms", 250),
                "off_ms": gpio_cfg.get("unknown_buzzer_off_ms", 250),
            },
        }

        # Callbacks (set by main.py)
        self.on_record_toggle: Callable[[], None] | None = None
        self.on_pause_5m: Callable[[], None] | None = None
        self.on_field_tag: Callable[[str], None] | None = None

        self._has_gpio = False
        self._gpio = None
        self._alarm_active = False
        self._strobe_active = False
        self._record_led_active = False
        self._stop_event = threading.Event()
        self._alarm_end_time: float = 0
        self._active_alert_level = "UNKNOWN_ALERT"
        self._strobe_thread: threading.Thread | None = None
        self._buzzer_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._button_events = deque(maxlen=30)

        self._init_gpio()

    def _init_gpio(self):
        if not self.enabled:
            logger.info("GPIO disabled in config")
            return
        try:
            import RPi.GPIO as GPIO

            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # Outputs
            for pin in [
                self.alert_pin,
                self.strobe_pin,
                self.record_led_pin,
            ]:
                if pin is None:
                    continue
                logger.info("GPIO setup OUTPUT pin %d (initial LOW)", pin)
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

            # Active-low button inputs: internal pull-up holds the pin HIGH;
            # pressing the button connects the pin to GND and triggers FALLING.
            for pin in [
                self.reset_pin,
                self.record_button_pin,
                self.pause_button_pin,
                *self.field_tag_button_pins.values(),
            ]:
                if pin is None:
                    continue
                logger.info("GPIO setup INPUT pin %d (pull-up)", pin)
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            # Interrupts
            if self.reset_pin is not None:
                GPIO.add_event_detect(
                    self.reset_pin,
                    GPIO.FALLING,
                    callback=self._on_reset_pressed,
                    bouncetime=300,
                )
            if self.record_button_pin is not None:
                GPIO.add_event_detect(
                    self.record_button_pin,
                    GPIO.FALLING,
                    callback=self._on_record_button,
                    bouncetime=500,
                )
            if self.pause_button_pin is not None:
                GPIO.add_event_detect(
                    self.pause_button_pin,
                    GPIO.FALLING,
                    callback=self._on_pause_button,
                    bouncetime=500,
                )
            for tag, pin in self.field_tag_button_pins.items():
                if pin is None:
                    continue
                GPIO.add_event_detect(
                    pin,
                    GPIO.FALLING,
                    callback=lambda channel, tag=tag: self._on_field_tag_button(
                        tag, channel
                    ),
                    bouncetime=500,
                )

            self._has_gpio = True
            logger.info(
                "GPIO initialized: ALERT=%s STROBE=%s RESET=%s "
                "REC_LED=%s REC_BTN=%s PAUSE_BTN=%s FIELD_TAGS=%s",
                self.alert_pin,
                self.strobe_pin,
                self.reset_pin,
                self.record_led_pin,
                self.record_button_pin,
                self.pause_button_pin,
                self.field_tag_button_pins,
            )
        except Exception as e:
            logger.error("GPIO init FAILED: %s", e)
            self._gpio = None
            self._has_gpio = False
            require_gpio = os.getenv("REQUIRE_GPIO", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if require_gpio:
                raise RuntimeError(
                    "GPIO is enabled in config but RPi.GPIO failed to "
                    f"initialize: {e}"
                ) from e
            logger.warning(
                "Continuing with mock GPIO; set REQUIRE_GPIO=true to fail hard"
            )

    # ------------------------------------------------------------------
    # Alarm API
    # ------------------------------------------------------------------

    def trigger_alarm(self, alert_level: str = "UNKNOWN_ALERT") -> bool:
        with self._lock:
            self._active_alert_level = alert_level or "UNKNOWN_ALERT"
            if self._alarm_active:
                self._alarm_end_time = time.time() + (
                    self.alert_duration_ms / 1000.0
                )
                return True
            self._alarm_active = True
            self._alarm_end_time = time.time() + (
                self.alert_duration_ms / 1000.0
            )

        logger.warning(
            "GPIO %s TRIGGERED (pin %s)", self._active_alert_level, self.alert_pin
        )
        self._start_buzzer()
        # self._start_strobe()
        threading.Thread(target=self._auto_clear_loop, daemon=True).start()
        return True

    def clear_alarm(self):
        with self._lock:
            if not self._alarm_active:
                return
            self._alarm_active = False
            self._alarm_end_time = 0
        self._write_pin(self.alert_pin, False)
        self._stop_strobe()
        logger.info("GPIO alert cleared")

    @property
    def is_alarming(self) -> bool:
        return self._alarm_active

    # ------------------------------------------------------------------
    # Recording LED
    # ------------------------------------------------------------------

    def set_record_led(self, on: bool):
        self._record_led_active = on
        self._write_pin(self.record_led_pin, on)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_reset_pressed(self, channel):
        logger.info("Reset button pressed (GPIO %d)", channel)
        self._record_button_event("reset", channel)
        self.clear_alarm()

    def _on_record_button(self, channel):
        logger.info("Record button pressed (GPIO %d)", channel)
        self._record_button_event("record", channel)
        if self.on_record_toggle:
            self.on_record_toggle()

    def _on_pause_button(self, channel):
        logger.info("Pause button pressed (GPIO %d)", channel)
        self._record_button_event("pause", channel)
        if self.on_pause_5m:
            self.on_pause_5m()

    def _on_field_tag_button(self, tag: str, channel):
        logger.info("Field tag %s button pressed (GPIO %d)", tag, channel)
        self._record_button_event(tag, channel)
        if self.on_field_tag:
            self.on_field_tag(tag)

    def _record_button_event(self, name: str, channel: int):
        self._button_events.append(
            {
                "name": name,
                "gpio": int(channel),
                "timestamp": time.time(),
            }
        )

    # ------------------------------------------------------------------
    # Strobe
    # ------------------------------------------------------------------

    def _start_strobe(self):
        with self._lock:
            if self._strobe_active:
                return
            self._strobe_active = True
        self._strobe_thread = threading.Thread(
            target=self._strobe_loop, daemon=True
        )
        self._strobe_thread.start()

    def _stop_strobe(self):
        with self._lock:
            self._strobe_active = False
        self._write_pin(self.strobe_pin, False)

    def _start_buzzer(self):
        if self._buzzer_thread and self._buzzer_thread.is_alive():
            return
        self._buzzer_thread = threading.Thread(
            target=self._buzzer_loop, daemon=True
        )
        self._buzzer_thread.start()

    def _buzzer_loop(self):
        while self._alarm_active and not self._stop_event.is_set():
            with self._lock:
                level = self._active_alert_level
            pattern = self.buzzer_patterns.get(
                level, self.buzzer_patterns["UNKNOWN_ALERT"]
            )
            on_s = max(0.02, float(pattern["on_ms"]) / 1000.0)
            off_s = max(0.02, float(pattern["off_ms"]) / 1000.0)
            self._write_pin(self.alert_pin, True)
            time.sleep(on_s)
            self._write_pin(self.alert_pin, False)
            time.sleep(off_s)

    def _strobe_loop(self):
        interval = self.pulse_interval_ms / 1000.0
        while self._strobe_active and not self._stop_event.is_set():
            self._write_pin(self.strobe_pin, True)
            time.sleep(interval)
            if not self._strobe_active:
                break
            self._write_pin(self.strobe_pin, False)
            time.sleep(interval)

    def _auto_clear_loop(self):
        while self._alarm_active and not self._stop_event.is_set():
            remaining = self._alarm_end_time - time.time()
            if remaining <= 0:
                self.clear_alarm()
                break
            time.sleep(min(remaining, 1.0))

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _write_pin(self, pin: int, state: bool):
        if pin is None:
            return
        if self._has_gpio and self._gpio:
            try:
                value = self._gpio.HIGH if state else self._gpio.LOW
                self._gpio.output(pin, value)
                logger.debug("GPIO pin %d → %s", pin, "HIGH" if state else "LOW")
            except Exception as e:
                logger.error(
                    "GPIO write FAILED pin %d → %s: %s",
                    pin,
                    "HIGH" if state else "LOW",
                    e,
                )
        else:
            logger.debug(
                "GPIO mock write pin %d → %s",
                pin,
                "HIGH" if state else "LOW",
            )

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "has_gpio": self._has_gpio,
            "alarming": self._alarm_active,
            "record_led": self._record_led_active,
            "alert_pin": self.alert_pin,
            "strobe_pin": self.strobe_pin,
            "reset_pin": self.reset_pin,
            "record_led_pin": self.record_led_pin,
            "record_button_pin": self.record_button_pin,
            "pause_button_pin": self.pause_button_pin,
            "field_tag_button_pins": self.field_tag_button_pins,
            "button_events": list(self._button_events),
            "active_alert_level": self._active_alert_level,
        }

    def cleanup(self):
        self.clear_alarm()
        self.set_record_led(False)
        self._stop_event.set()
        if self._has_gpio and self._gpio:
            try:
                self._gpio.cleanup()
                logger.info("GPIO cleaned up")
            except Exception:
                pass
