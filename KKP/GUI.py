import numpy as np
from BrukerPS import BrukerPS
from KepcoPS import KepcoPS
import SR850LIA as SR850
import time

from pymeasure.experiment import Parameter, FloatParameter, IntegerParameter
from pymeasure.experiment import Procedure
from pymeasure.experiment import Results, Worker

from pymeasure.experiment import Experiment



class AngularScan(Procedure):

    angle_start = FloatParameter("Start angle", units="deg", minimum=0, maximum=360, default=0)
    angle_stop  = FloatParameter("Stop angle", units="deg", minimum=0, maximum=360, default=360)
    angle_step  = FloatParameter("Angle step", units="deg", minimum=0.1, maximum=90, default=3)
    field_strength = FloatParameter("Field strength", units="mT", minimum=0, maximum=1000, default=100)

    inputs   = ["angle_start", "angle_stop", "angle_step", "field_strength"]
    displays = ["angle_start", "angle_stop", "angle_step", "fieild_strength"]
    DATA_COLUMNS = ["angle", "V1w", "V2wx", "V2wy"]   # ← REQUIRED FIX

    def startup(self):
        print("Starting FAKE measurement…")
        try:
            self.lia = SR850("GPIB::8")
            self.ps1 = KepcoPS("GPIB::1")
            self.ps2 = BrukerPS("COM3")
            self.hardware_available = True
        except Exception as e:
            print("Hardware not available → using fake data:", e)
            self.hardware_available = False


    def execute(self):
        for angle in np.arange(self.angle_start, self.angle_stop + self.angle_step, self.angle_step):
            print("Angle", angle)

            try:
                current_Kepco, current_Bruker = self.calculate_currents(angle, self.field_strength)
                print("Setting currents:", current_Kepco, current_Bruker)
                self.set_currents(current_Kepco, current_Bruker)
            except:
                self.current_Kepco, self.current_Bruker = 0, 0

            # ---- FAKE DATA ----
            if self.hardware_available:
                V1w = self.lia.read_1w()
                V2wx, V2wy = self.lia.read_2w()
            else:
                V1w = 0
                V2wx = 0
                V2wy = 0

            # Emit data for GUI and CSV
            self.emit("results", {
                "angle": angle,
                "V1w": V1w,
                "V2wx": V2wx,
                "V2wy": V2wy

            })

            # Optional: progress bar
            progress = (angle - self.angle_start) / (self.angle_stop - self.angle_start) * 100
            self.emit("progress", progress)
            
            time.sleep(0.1)

            if self.should_stop():
                break

    def shutdown(self):
        try:
            self.ps1.output = False
            self.ps2.output = False
        except:
            pass


from pymeasure.display.windows import ManagedWindow
import sys
from pymeasure.experiment import Results

from pymeasure.display.windows import ManagedWindow


class MainWindow(ManagedWindow):
    def __init__(self):
        super().__init__(

            procedure_class=AngularScan,
            inputs=["angle_start", "angle_stop", "angle_step", "field_strength"],
            displays=["angle_start", "angle_stop", "angle_step", "field_strength"],  # FIXED
            x_axis="angle",
            y_axis="V1w",
        )
        print("DATA_COLUMNS seen by GUI:", AngularScan.DATA_COLUMNS)

        self.setWindowTitle("2ω Harmonic Hall Angular Scan")


def queue(self, procedure=None):
    if procedure is None:
        procedure = self.make_procedure()

    results = Results(procedure, "fake_scan.csv")

    # This creates an Experiment the correct way for the current PyMeasure version
    experiment = self.new_experiment(results)

    self.manager.queue(experiment)



if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())