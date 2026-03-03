"""
harmonic_measurement.py
Perform in-plane angular 2nd-harmonic Hall scans while controlling:
 - SR830 (lock-in) to read first/second harmonic (via SNAP? or specific channels)
 - Magnet controller (set magnitude and angle)  -> adapt vendor-specific commands
 - Voltage/Current source (set drive amplitude) -> adapt vendor-specific commands
VISA = Virtual Instrument Software Architecture
Adapt the VISA resource strings below for your instruments (use `pyvisa.ResourceManager().list_resources()` to find them).
"""

import time
import csv
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import pyvisa

# -------------------------
# User configuration
# -------------------------
# VISA resource strings (example placeholders)
SR830_RESOURCE = "GPIB0::10::INSTR"           # change to your SR830 address
MAGNET_RESOURCE = "ASRL1::INSTR"              # serial port for magnet controller (example)
SOURCE_RESOURCE = "GPIB0::5::INSTR"           # function generator or source-meter address

# Measurement parameters
angles_deg = np.linspace(0, 360, 37)          # angles to step (deg)
field_values_T = [0.1]                        # list of applied external field magnitudes (Tesla) - adapt units
drive_voltages = [0.1, 0.2]                   # drive amplitudes (example) — units depend on your source (V or A)
averages_per_point = 5                        # average this many reads per point
stabilization_wait_s = 2.0                    # wait after setting field/voltage (increase for large magnets)
field_tolerance = 0.001                       # tolerance for field settling (T). If not available, increase wait.

output_csv = "harmonic_scan_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M%S"))

# -------------------------
# Helper: VISA open
# -------------------------
rm = pyvisa.ResourceManager()

# -------------------------
# SR830 wrapper
# -------------------------
from stanford_SR830_modified import liaSR830

# -------------------------
# Magnet controller wrapper (placeholder)
# -------------------------
# import ...

# -------------------------
# Voltage/Current source wrapper (placeholder)
# -------------------------
class Source:
    def __init__(self, resource):
        self.res = rm.open_resource(resource)
        self.res.timeout = 5000
        self.res.write_termination = '\n'
        self.res.read_termination = '\n'
    def set_voltage(self, volts):
        # Example SCPI for an arbitrary waveform generator or source-meter:
        # For Keithley 2400 use: "VOLT {volts}"
        # For an AWG you may need a different command.
        self.res.write(f"VOLT {float(volts)}")
    def enable_output(self, on=True):
        self.res.write("OUTP ON" if on else "OUTP OFF")
    def close(self):
        try:
            self.res.close()
        except Exception:
            pass

# -------------------------
# Measurement routine
# -------------------------
def perform_scan():
    # Open instruments
    print("Opening instruments...")
    sr = liaSR830(SR830_RESOURCE)
    mg = MagnetController(MAGNET_RESOURCE)
    src = Source(SOURCE_RESOURCE)

    # Configure SR830 as needed (example)
    # Choose detection harmonic (1 or 2). For second harmonic detection, set HARM 2.
    sr.SetRefHarm(2)   # we want 2nd harmonic reading
    # Choose SNAP parameters to read: X (1) and Y (2) or R (3)
    snap_params = ('X','Y')   # adapt: could be ('R',) or ('X','Y','Aux1') etc.

    # Create output CSV and write header
    header = ['timestamp', 'angle_deg', 'field_T', 'drive_V', 'avg_X', 'avg_Y', 'raw_samples']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    try:
        # Ensure source output is off initially
        src.enable_output(False)

        for drive in drive_voltages:
            print(f"Setting drive to {drive} ...")
            src.set_voltage(drive)
            src.enable_output(True)
            time.sleep(0.5)  # short settle after enabling source

            for field in field_values_T:
                # Optional: for small magnets this could be quick; for superconducting magnets allow long settle
                for angle in angles_deg:
                    print(f"Drive={drive:.3f}  Field={field:.3f}T  Angle={angle:.1f} deg -> setting magnet...")
                    mg.set_field_and_angle(field, angle)

                    # Wait for field stabilization. If controller supports readback, poll until stable.
                    time.sleep(stabilization_wait_s)

                    # Optionally poll get_field = mg.query_field() and loop until |get_field - field| < field_tolerance

                    # Acquire N averages
                    samples = []
                    for i in range(averages_per_point):
                        try:
                            vals = sr.snapshot(*snap_params)   # returns list of floats matching params
                        except Exception as e:
                            print("SR read error:", e)
                            vals = [np.nan]*len(snap_params)
                        samples.append(vals)
                        time.sleep(0.05)   # short delay between reads

                    arr = np.array(samples, dtype=float)
                    mean_vals = np.nanmean(arr, axis=0)
                    # For X,Y snapshot the mean_vals[0] is X, mean_vals[1] is Y
                    avg_x = mean_vals[0] if len(mean_vals) > 0 else np.nan
                    avg_y = mean_vals[1] if len(mean_vals) > 1 else np.nan

                    timestamp = datetime.now().isoformat()
                    # Save with raw samples as string for traceability
                    with open(output_csv, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, angle, field, drive, avg_x, avg_y, repr(arr.tolist())])

                    print(f"  -> saved: X={avg_x:.6e}, Y={avg_y:.6e}")

            # After finishing drive amplitude, optionally turn source off briefly
            src.enable_output(False)

    except KeyboardInterrupt:
        print("Measurement interrupted by user, shutting down safely...")
    except Exception as e:
        print("Error during measurement:", e)
        raise
    finally:
        # safe shutdown
        print("Shutting down: turning source off and closing instruments...")
        try:
            src.enable_output(False)
        except Exception:
            pass
        sr.close()
        mg.close()
        src.close()
        print("Done. Results saved to:", output_csv)

if __name__ == "__main__":
    perform_scan()
