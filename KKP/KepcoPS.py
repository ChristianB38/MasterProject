import time
import pyvisa


class KepcoPS:
    def __init__(self, address="GPIB0::5::INSTR"):
        rm = pyvisa.ResourceManager()
        self.inst = rm.open_resource(address)
        self.inst.timeout = 3000
        self.inst.write_termination = '\n'
        self.inst.read_termination = '\n'

        print("Connected to:", self.idn)

        # Safety: turn OFF output before enabling
        if self.output_state:
            print("Output was ON — turning OFF for safety.")
            self.output_state = False

        self.reset()
        self.clear()

        print("Initializing KEPCO in CC mode...")
        self.reset()
        self.clear()

        self.inst.write("FUNC CURR")

        print("Enabling output...")
        self.output_state = True
        self.sleep_with_countdown(10)
        print("KEPCO should be ready to rumble!")

    def sleep_with_countdown(self, seconds):
        for i in range(seconds):
            print(f"Waiting... {seconds - i} s remaining", end="\r")
            time.sleep(1)
        print(" " * 40, end="\r")
        
    # ---------------------------------------------------------------
    # Convenience Properties
    # ---------------------------------------------------------------

    @property
    def idn(self):
        """Return identification string."""
        return self.inst.query("*IDN?").strip()

    @property
    def output_state(self):
        """Return True if output is ON."""
        return self.inst.query("OUTP?").strip() == "1"

    @output_state.setter
    def output_state(self, value: bool):
        """Enable or disable output."""
        cmd = "ON" if value else "OFF"
        self.inst.write(f"OUTP {cmd}")
        time.sleep(0.2)

    # ---------------------------------------------------------------
    # Reset and Clear
    # ---------------------------------------------------------------

    def reset(self):
        self.inst.write("*RST")

    def clear(self):
        self.inst.write("*CLS")

    # ---------------------------------------------------------------
    # Current setpoint (A) — CONSTANT CURRENT MODE
    # ---------------------------------------------------------------

    @property
    def current(self):
        """Return the *measured* output current."""
        return float(self.inst.query("MEAS:CURR?"))

    @current.setter
    def current(self, value):
        """
        Set the *target* current.
        The device stays in CC mode.
        """
        self.inst.write("FUNC CURR")  # make sure CC mode is active
        self.inst.write(f"CURR {value}")
        print(f"Set current to {value} A")
        self.sleep_with_countdown(5)

    # ---------------------------------------------------------------
    # Voltage Compliance (V)
    # ---------------------------------------------------------------

    @property
    def voltage_limit(self):
        """Voltage compliance limit."""
        return float(self.inst.query("VOLT?"))

    @voltage_limit.setter
    def voltage_limit(self, value):
        self.inst.write(f"VOLT {value}")
        print(f"Set voltage limit to {value} V")
        time.sleep(0.1)

    # ---------------------------------------------------------------
    # Measurement
    # ---------------------------------------------------------------

    @property
    def measured_voltage(self):
        return float(self.inst.query("MEAS:VOLT?"))

    # ---------------------------------------------------------------
    # Startup and Shutdown sequences
    # ---------------------------------------------------------------

    def shutdown(self):
        """Safe output disable."""
        print("Shutting down…")
        self.output_state = False
        self.sleep_with_countdown(10)

        i = self.current
        v = self.measured_voltage

        print(f"After shutdown: I = {i} A, V = {v} V")
        self.close()

    # ---------------------------------------------------------------
    # Close session
    # ---------------------------------------------------------------

    def close(self):
        self.inst.close()
        print("KEPCO connection closed.")
