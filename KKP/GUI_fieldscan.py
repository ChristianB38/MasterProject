# Todos: 
# - Check magnet configuration, which one is x and which one is y, and for neg/pos current is the field in which direction?
# - Learn to work with the LIA and how to connect
# - Create function to measure from LIA, (1st HH, V1w = float(self.lia.ask("OUTP? 3"))     # R (1ω)V2wx = float(self.lia.ask("OUTP? 5"))    # X (2ω)V2wy = float(self.lia.ask("OUTP? 6"))    # Y (2ω))


import numpy as np
from BrukerPS import BrukerPS
from KepcoPS import KepcoPS
import SR850LIA as SR850
from pymeasure.instruments.srs import SR830

import time
import sys 
from pymeasure.experiment import Parameter, FloatParameter, IntegerParameter, ListParameter
from pymeasure.experiment import Procedure
from pymeasure.experiment import Results, Worker
from pymeasure.experiment import Experiment
from collections import OrderedDict

from PyQt5.QtWidgets import QApplication


class FieldScan(Procedure):
    # Creating inputs for field scan parameters
    field_start = FloatParameter("Field start", units="mT", minimum=0, maximum=100, default=0)
    field_stop  = FloatParameter("Field stop", units="mT", minimum=0, maximum=100, default=100)
    field_step  = FloatParameter("Field step", units="mT", minimum=0.1, maximum=90, default=1)
    angle  = FloatParameter("angle", units="deg", minimum=0, maximum=360, default=0)

    # Creating inputs for LIA parameters
    voltage = FloatParameter("Sine voltage (Vrms)", units="V", minimum=4e-3, maximum=5, default=1)
    frequency = FloatParameter("Frequency", units="Hz", minimum=1e-3, maximum=102000, default=133)
    LIA_sensitivity = ListParameter("Sensitivity", units="V", choices=[
        2e-9, 5e-9,
        1e-8, 2e-8, 5e-8,
        1e-7, 2e-7, 5e-7,
        1e-6, 2e-6, 5e-6,
        1e-5, 2e-5, 5e-5,
        1e-4, 2e-4, 5e-4,
        1e-3, 2e-3, 5e-3,
        1e-2, 2e-2, 5e-2,
        1e-1, 2e-1, 5e-1,
        1.0
    ], default=1e-6)    
    LIA_timeconstant = FloatParameter("Time Constant", units="s", minimum=1e-5, maximum=30, default=0.3)
    Wait_time = FloatParameter("Wait time", units="s", minimum=0, maximum=100, default=2)

    # Creating inputs for device parameters
    resistance_sample = FloatParameter("Sample resistance", units="Ohm", minimum=0, maximum=1e6, default=100)
    resistance_xx = FloatParameter("Rxx resistance", units="Ohm", minimum=0, maximum=1e6, default=10)

    # Grouping parameters for better GUI organization
    parameter_groups = OrderedDict({
        "Field parameters": ["field_start", "field_stop", "field_step", "angle"],
        "LIA parameters": ["voltage", "frequency", "LIA_sensitivity", "LIA_timeconstant", "Wait_time"],
        "Device parameters": ["resistance_sample", "resistance_xx"]
    })

    # Creating lists for inputs and displays in the GUI and saved columns in the datafile
    inputs   = ["angle", "field_start", "field_stop", "field_step", 
                "voltage", "frequency", "LIA_sensitivity", "LIA_timeconstant", "Wait_time",
                "resistance_sample", "resistance_xx"]
    displays = ["angle", "field_start", "field_stop", "field_step", "voltage"]
    DATA_COLUMNS = ["Current_bruker", "Current_Kepco", "Angle", "Field", "V1w", "V2wX", "V2wY"]   

    def sleep_with_countdown(self, seconds):
        for i in range(seconds):
            print(f"Waiting... {seconds - i} s remaining", end="\r")
            time.sleep(1)
        print(" " * 40, end="\r")

    def startup(self):
        print("Starting FAKE measurement…")
        try:
            # Initialize LIA
            self.lia = SR830("GPIB::8")
            self.lia.sine_voltage = self.voltage / (2 * np.sqrt(2) )   # RMS to Peak conversion
            self.lia.frequency = self.frequency
            self.lia.sensitivity = self.LIA_sensitivity
            self.lia.time_constant = self.LIA_timeconstant

            # Check if connected
            print(self.lia.x)          # Read X value
            print(self.lia.frequency)  # Read frequency
            print(self.lia.sensitivity) # Read sensitivity
            print(self.lia.time_constant) # Read time constant
            
            # Initialize Power Supplies
            self.ps1 = KepcoPS("GPIB::1")
            self.ps2 = BrukerPS("COM3")
            self.hardware_available = True
        except Exception as e:
            print("Hardware not available → using fake data:", e)
            self.hardware_available = False

    def calculate_current(self, angle, Bmag):
        # Calibration constants (mT/A)
        k_x = 9.24633  
        k_y = 3.84322

        if type(angle) == float or type(angle) == int:
            theta = np.deg2rad(angle)
            Bx = Bmag * np.cos(theta)
            By = Bmag * np.sin(theta)

            return Bx / k_x, By / k_y
    
    def set_currents(self, current_Kepco, current_Bruker):
        if self.hardware_available:
            self.ps1.current = current_Kepco
            self.sleep_with_countdown(1)
            self.ps2.current = current_Bruker
            self.sleep_with_countdown(1)
    
    def measure_LIA(self):
        X = self.lia.x
        Y = self.lia.y
        return X, Y

    def execute(self):
        fields = np.arange(self.field_start, self.field_stop + self.field_step, self.field_step)
        for i, field in enumerate(fields):
            progress = i / len(fields) * 100
            self.emit("progress", progress)

            self.field = field
            print("Field", field)

            try:
                current_Kepco, current_Bruker = self.calculate_current(self.angle, self.field)
                print("Setting currents:", current_Kepco, current_Bruker)

                if current_Kepco < 10 and current_Bruker < 10:
                    self.set_currents(current_Kepco, current_Bruker)
            except:
                self.current_Kepco, self.current_Bruker = 0, 0

            # ---- FAKE DATA ----
            if self.hardware_available:
                self.sleep_with_countdown(int(self.Wait_time))
                X, Y = self.measure_LIA()
            else:
                current_Kepco, current_Bruker = 0, 0
                V1w = np.cos(field)
                V2wX = 0
                V2wY = 0

            # Emit data for GUI and CSV
            self.emit("results", {
                "Current_Kepco": current_Kepco,
                "Current_bruker": current_Bruker,
                "Angle": self.angle,
                "Field": field,
                "V1w": V1w,
                "V2wX": V2wX,
                "V2wY": V2wY
            })
            self.sleep_with_countdown(1)

            if self.should_stop():
                break

    def shutdown(self):
        self.set_currents(0, 0)
        self.ps1.shutdown()
        self.ps2.shutdown()
        return



from pymeasure.display.windows import ManagedWindow
import sys
from pymeasure.experiment import Results

from pymeasure.display.windows import ManagedWindow


class MainWindow(ManagedWindow):
    def __init__(self):
        super().__init__(
            procedure_class=FieldScan,
            inputs = FieldScan.inputs,
            displays= FieldScan.displays,  # FIXED
            x_axis="Field",
            y_axis="V1w",
        )
        
        self.setWindowTitle("Field Scan")         # optional
        self.directory = r"C:\git\MasterProject\data"
        self.filename = "FieldScan_results"

        print("DATA_COLUMNS seen by GUI:", FieldScan.DATA_COLUMNS)
        return


def queue(self, procedure=None):
    if procedure is None:
        procedure = self.make_procedure()

    results = Results(procedure, "fake_scan.csv")

    # This creates an Experiment the correct way for the current PyMeasure version
    experiment = self.new_experiment(results)

    self.manager.queue(experiment)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    from pymeasure.display.windows import ManagedWindow
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())