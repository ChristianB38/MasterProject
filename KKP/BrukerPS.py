# Class Bruker Power Supply

import time
import re
import serial
import numpy as np


class BrukerPS:
    """
    PyMeasure-style instrument wrapper for the Bruker magnet power supply.
    Provides .current get/set and handles serial communication internally.
    """

    def __init__(self, port="COM1", baudrate=9600, timeout=1):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity=serial.PARITY_NONE,
            stopbits=1,
            timeout=timeout
        )
        print(f"Connected to Bruker PSU on {port}, communicating with Bruker...")
        self.send_cmd("RST=0")
        print("System status:")
        self.send_cmd("STA/")
        print("Set to remote mode:")
        self.send_cmd("REM/")
        print("Turn on power supply...")
        self.send_cmd("DCP=1")
        self.sleep_with_countdown(15)
        print("Bruker PSU should be ready to rumble!")

    # ---------------------------------------------------------
    # Utility functions
    # ---------------------------------------------------------
    def sleep_with_countdown(self, seconds):
        for i in range(seconds):
            print(f"Waiting... {seconds - i} s remaining", end="\r")
            time.sleep(1)
        print(" " * 40, end="\r")

    def send_cmd(self, cmd):
        """Send a command and return the string response."""
        self.ser.write((cmd + "\r").encode())
        time.sleep(0.2)
        resp = self.ser.read(200).decode(errors="ignore").strip()
        print(f"Sent: {cmd}  ->  Response: {resp}")
        return resp

    def extract_current(self, s):
        match = re.search(r"([-+]?\d*\.?\d+)\s*A", s)
        if match:
            return float(match.group(1))
        return None

    # ---------------------------------------------------------
    # PyMeasure-style CURRENT PROPERTY
    # ---------------------------------------------------------

    @property
    def current(self):
        """Read the actual output current (A)."""
        resp = self.send_cmd("CHN/")
        value = self.extract_current(resp)
        return value

    @current.setter
    def current(self, value):
        """Set the PSU output current (A)."""
        print(f"Setting Bruker PSU current to {value} A")
        self.send_cmd(f"CUR={value}")
        self.sleep_with_countdown(15)

        # Check if the current actually settled
        cur = self.current
        while np.round(cur, 2) != np.round(value, 2):
            print("Current deviated; retrying...")
            self.startup()
            self.send_cmd(f"CUR={value}")
            cur = self.current

        print("Current applied successfully.")

    # ---------------------------------------------------------
    # Commands for the device
    # ---------------------------------------------------------

    def check_startup(self):
        # Safety check
        dcp = self.send_cmd("DCP/")
        if dcp == "DCP/ 0":
            print("Bruker PSU not started up, starting up now...")
            self.send_cmd("DCP=1")
            self.sleep_with_countdown(15)

    def shutdown(self):
        print("Turning off Bruker PSU...")
        self.send_cmd("DCP=0")
        self.ser.close()
        print("Serial port closed.")

    def measure(self):
        print("Current:", self.send_cmd("CHN/"))
        print("Voltage:", self.send_cmd("CHV/"))
        print("Referenced current:", self.send_cmd("CUR/"))


