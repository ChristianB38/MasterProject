# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import re
import os 
import glob
import sys
from pathlib import Path
import itertools
import matplotlib.ticker as ticker
# STFMR Class 1.1
class HH:
    def __init__(self, folder=None, file=None, d_FM=None, d_NM=None, Ms=None, Width=None, Length=None, used_fields='Pos', Bk=0.1, sample_name=None):
        if folder is not None:
            self.folder = folder
        if file is not None:
            self.file = file

        self.pattern = re.compile(
            r"HI_(?P<phi>[+-]?\d+(?:\.\d+)?)A_"
            r"U_(?P<freq>[+-]?\d+(?:\.\d+)?)V_"
            r"HG_(?P<pow>[+-]?\d+(?:\.\d+)?)mT"
        )

        # Physical constants
        self.mu_B = 9.274009994e-24  # Bohr magneton [J/T]
        self.e = 1.602176634e-19     # elementary charge [C]
        self.mu_0 = 4 * np.pi * 1e-7 # vacuum permeability [H/m]
        self.gamma = 1.760859e11     # gyromagnetic ratio [rad/(s·T)]
        self.hbar = 6.626e-34

        # System settings
        self.d_FM = d_FM
        self.d_NM = d_NM
        self.Ms = Ms
        self.Width = Width
        self.Length = Length
        self.used_fields=used_fields
        self.Bk = Bk
        self.sample_name = sample_name

    def get_key(self, filepath):
        """Create a key for a file, which describes the most important parameters of the datafile. Distinghuishes between measurement with or without DC."""
        filename = Path(filepath).name

        match = self.pattern.search(filename)
        if match:
            (DC, Voltage, Field) = match.groups()
            return (DC, Voltage, Field)
        return None
    
    def read_header(self, filepath):
        """
        Reads the first 5 header lines of the measurement file and stores
        the parameters as instance attributes.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            # Read exactly the first 5 parameter lines
            lines = [f.readline().strip() for _ in range(4)]
        
        header_data = {}
        # Pattern: "Label: number unit"
        pattern = r"^(.*?):\s*([+-]?\d*\.?\d+)\s*(.*)$"

        for line in lines:
            m = re.match(pattern, line)
            if m:
                label, value, unit = m.groups()
                key = label.replace(" ", "_")
                header_data[key] = float(value)
                header_data[key + "_unit"] = unit
        return header_data

    def read_file(self, filepath):
        """Reads a measurement file and returns the data as a pandas DataFrame."""
        data = pd.read_csv(filepath, skiprows=5, sep='\s+')
        header_data = self.read_header(filepath)
        return data, header_data

    def plot_first_harmonic(self, data, header_data):
        plt.figure(figsize=(10, 6))
        plt.plot(data['Angle'], data['V1w'], label='First Harmonic')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('First Harmonic Hall Voltage (V)')
        plt.title(f'First Harmonic vs Angle, B = {header_data["Gausseter"]} {header_data["Gausseter_unit"]} and V = {header_data["Voltage_for_measurment"]} {header_data["Voltage_for_measurment_unit"]}')
        plt.legend()
        plt.show()

    def plot_second_harmonic(self, data, header_data, textbox=True):
        plt.figure(figsize=(10, 6))
        plt.plot(data['Angle'], data['V2w_x']*1e6, label='Second Harmonic X-signal', color='orange')
        plt.plot(data['Angle'], data['V2w_y']*1e6, label='Second Harmonic Y-signal', color='green')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Second Harmonic Hall Voltage (uV)')
        plt.title(f'Second Harmonic vs Angle, B = {header_data["Gausseter"]} {header_data["Gausseter_unit"]} and V = {header_data["Voltage_for_measurment"]} {header_data["Voltage_for_measurment_unit"]}')
        plt.legend()
        plt.show()

    def plot_both_harmonics(self, data, header_data):
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # ------------------------------
        # Subplot 1 — First harmonic
        # ------------------------------
        axs[0].plot(data['Angle'], data['V1w'], label='First Harmonic')
        axs[0].set_ylabel('First Harmonic Hall Voltage (V)')
        axs[0].set_title(
            f'First Harmonic vs Angle, '
            f'B = {header_data["Gausseter"]} {header_data["Gausseter_unit"]}, '
            f'V = {header_data["Voltage_for_measurment"]} {header_data["Voltage_for_measurment_unit"]}'
        )        
        axs[0].legend()

        # ------------------------------
        # Subplot 2 — Second harmonic
        # ------------------------------
        axs[1].plot(data['Angle'], data['V2w_x']*1e6,
                    label='Second Harmonic X-signal', color='orange')
        axs[1].plot(data['Angle'], data['V2w_y']*1e6,
                    label='Second Harmonic Y-signal', color='green')

        axs[1].set_xlabel('Angle (degrees)')
        axs[1].set_ylabel('Second Harmonic Hall Voltage (µV)')
        axs[1].set_title(
            f'Second Harmonic vs Angle, '
            f'B = {header_data["Gausseter"]} {header_data["Gausseter_unit"]}, '
            f'V = {header_data["Voltage_for_measurment"]} {header_data["Voltage_for_measurment_unit"]}'
        )
        axs[1].legend()

        # Improve spacing
        plt.tight_layout()
        plt.show()

    def fit_2nd_harm(self, data, header_data, plot=False, cutoff=1):

        # Convert angles to radians for fitting
        angles_rad = np.deg2rad(data['Angle'])
        popt, pcov = curve_fit(self.angular_dependence,
                            angles_rad[cutoff:-cutoff],
                            data['V2w_y'][cutoff:-cutoff],)

        # Create phi in *degrees* but also phi_rad for evaluation
        phi = self.create_fit_array(np.min(data['Angle']), np.max(data['Angle']), num_points=200)
        phi_rad = np.deg2rad(phi)
        if plot:
            fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            # Plot raw data (degrees)
            axs[0].plot(data['Angle'], data['V2w_y'] * 1e6,
                        label='Second Harmonic Y-signal',
                        color='orange')

            # Plot fit (converted to radians)
            axs[0].plot(phi, self.angular_dependence(phi_rad, *popt) * 1e6,
                        label='Fit', color='red', linestyle='--')

            # Cosine component — also use radians!
            axs[0].plot(phi, popt[0] * np.cos(phi_rad + popt[3]) * 1e6 + popt[2] * 1e6,
                        label='Cosine Component', color='blue', linestyle=':')

            axs[0].set_ylabel('Second Harmonic Hall Voltage (V)')
            axs[0].set_title(
                f'Second Harmonic vs Angle, '
                f'B = {header_data["Gausseter"]} {header_data["Gausseter_unit"]}, '
                f'V = {header_data["Voltage_for_measurment"]} {header_data["Voltage_for_measurment_unit"]}'
            )        
            axs[0].legend()

            # ------------------------------
            # Subplot 2 — Second harmonic
            # ------------------------------
            dep = (data['V2w_y'] - popt[0]*np.cos(angles_rad))*1e6

            axs[1].plot(data['Angle'], dep,
                        label='Only ... depenence', color='orange')
            axs[1].plot(phi, (popt[1]*(2*np.cos(phi_rad+popt[3])**3 - np.cos(phi_rad+popt[3])) + popt[2])*1e6,
                        label='Fit', color='red', linestyle='--')
            axs[1].set_xlabel('Angle (degrees)')
            axs[1].set_ylabel('Second Harmonic Hall Voltage (µV)')
            axs[1].set_title(
                f'Second Harmonic vs Angle, '
                f'B = {header_data["Gausseter"]} {header_data["Gausseter_unit"]}, '
                f'V = {header_data["Voltage_for_measurment"]} {header_data["Voltage_for_measurment_unit"]}'
            )
            axs[1].legend()

            # Improve spacing
            plt.tight_layout()
            plt.show()
        return popt, pcov

    def angular_dependence(self, phi, A, B, C, phi0):
        return A * np.cos(phi + phi0) + B * (2*np.cos(phi + phi0)**3 - np.cos(phi + phi0)) + C
    
    def plot_field_dependence(self):
        # Step 1: Group by applied voltage
        groups = {}
        for f in os.listdir(self.folder):
            fpath = os.path.join(self.folder, f)
            _, header = self.read_file(fpath)
            V = header["Voltage_for_measurment"]
            groups.setdefault(V, []).append(f)

        fig, ax = plt.subplots(2, 1, figsize=(12,16))

        # Step 2: For each voltage, fit all files and plot results
        for V, files in groups.items():
            print(f'Processing voltage: {V} V with files: {files}')

            popts_list = []
            fields_FL = []
            fields_DL = []

            for file in files:
                data, header_data = self.read_file(os.path.join(self.folder, file))
                popt, pocv = self.fit_2nd_harm(data, header_data)
                popts_list.append(popt)  # mT → T
                fields_FL.append(header_data["Gausseter"] / 1000)
                fields_DL.append(header_data["Gausseter"] / 1000 + self.Bk)

            ax[0].plot(1/np.array(fields_FL), np.array(popts_list)[:, 0], marker='o', label=f'V = {V} V')
            ax[1].plot(1/np.array(fields_FL), np.array(popts_list)[:, 1], marker='o', label=f'V = {V} V')

        ax[0].set_xlabel('1/(Bext + Bk) ($T^{-1}$)')
        ax[0].set_ylabel('Coefficient A = $\\left[ \\frac{R_{AHE}}{2} B_{AD} + T \\right]$(V)')
        ax[0].set_title('Field Dependence of cos($\\phi$) ')
        ax[1].set_xlabel('1/Bext ($T^{-1}$)')
        ax[1].set_ylabel('Coefficient B = \\left[ 2 $R_{PHE}(B_{FL} + B_{Oe})$ (V) \\right]')
        ax[1].set_title('Field Dependence of ($2cos^3(\\phi) - cos(\\phi)$) ')
        ax[0].legend()
        ax[1].legend()

    def extract_AD_and_FL_and_thermal(self, filelist, plot=True, V=None, color='blue', zerofield_data=False, plot_fits=False):
        """Read a folder, extract the angular dependence coefficients A and B"""
        results = []  # list of dicts: {Bext, A, B}
        # print(zerofield_data)
        # ---- STEP 1: Fit angular dependence for each field ----
        for file in filelist:
            filepath = os.path.join(self.folder, file)
            try:
                data, header = self.read_file(filepath)
                if abs(header["Gausseter"]) < 15:
                    # print("Plot of zerofielddata")
                    # popt, pcov = self.fit_2nd_harm(plot=plot_fits,data=data, header_data=header)
                    continue
                phi_rad = np.deg2rad(data["Angle"])
                # popt, pcov = self.fit_2nd_harm(plot=plot_fits,data=data, header_data=header)

                if zerofield_data is True:
                    try:
                        data["V2w_y"] -= zerofield_data
                        # print(f'Applied zerofield correction to file {file}')
                    except:
                        pass


                # Fit model: A*cos(phi + phi0) + B*(2*cos^3(phi + phi0) - cos(phi + phi0)) + C
                popt, pcov = self.fit_2nd_harm(plot=plot_fits,data=data, header_data=header)
                A_fit, B_fit, C, phi0 = popt
                Bext = header["Gausseter"] / 1000  # mT → T
                
                # self.fit_2nd_harm(data, header, plot=True)
                results.append({
                    "file": file,
                    "Bext": Bext,
                    "A": A_fit,
                    "A_err": np.sqrt(pcov[0,0]),
                    "B": B_fit,
                    "B_err": np.sqrt(pcov[1,1]),
                    'Size': len(data['V2w_y'])
                })
            except:
                print(f'Failed to extract data from {file}, skipping file.')
                    
        # ---- STEP 2: AD + thermal extraction ----
        results = pd.DataFrame(results)

        results["x"] = abs(1 / (results["Bext"] + self.Bk))
        results["x_FL"] = abs(1 / results["Bext"])

       # Suppose you have uncertainties in A: sigma_A
        x = results["x"]
        y = results["A"]
        sigma_y = results["A_err"]     # your A errors

        p, cov = np.polyfit(x, y, 1, w=1/sigma_y, cov=True)

        S, C = p                   # slope, intercept
        sigma_S, sigma_C = np.sqrt(np.diag(cov))    # uncertainties

        x_FL = results["x_FL"].values
        y_FL = results["B"].values
        
        sigma_y_FL = results["B_err"].values     # your B errors

        p_FL, cov_FL = np.polyfit(x_FL, y_FL, 1, w=1/sigma_y_FL, cov=True)
        sigma_S_FL, sigma_C_FL = np.sqrt(np.diag(cov_FL))

        # Plot
        if plot is not None:
            plot[0].errorbar(
                x, y, yerr=sigma_y,               # Ensure a marker shape is defined
                ms=4,                    # Increase marker size (default is 6)
                fmt='o',
                markerfacecolor='black', 
                markeredgewidth=1,     # Thicker border around the point
                label=f'Data Points V = {V}', 
                color=color,
                elinewidth=1,            # Thicker error bar lines
                capsize=3,               # Width of the horizontal caps
            )            
            fit_data_x = self.create_fit_array(min(x), max(x), num_points=100)
            fit_data_x_FL = self.create_fit_array(min(x_FL), max(x_FL), num_points=100)

            plot[0].plot(fit_data_x, S*fit_data_x + C, linestyle='--', color=color)
            # plot[0].set_xlim(left=-20)
            plot[1].errorbar(x_FL, y_FL, yerr=sigma_y_FL, fmt='o', label='Data Points', color=color)
            plot[1].plot(fit_data_x_FL, p_FL[0]*fit_data_x_FL + p_FL[1], linestyle='--', color=color)
            # plot[1].set_xlim(left=-20)
        # Return full dataset + extracted parameters
        return {
            "results_per_field": results,
            "AD_SOT": S,
            "AD_SOT_err": sigma_S,
            "thermal": C,
            "thermal_err": sigma_C,
            "FL_SOT": p_FL[0],     # == 2Rphe x (Bfl + Boe) / Bext
            "FL_SOT_err": sigma_S_FL,    
        }

    def extract_AD_and_FL_and_thermal_for_voltages(self, zerofield_correction=True, plot_fits=False):
        # group by applied voltage
        groups = {}
        zerofield_data = {}

        for f in os.listdir(self.folder):
            if f.lower().endswith(".txt"):
                fpath = os.path.join(self.folder, f)
                data, header = self.read_file(fpath)
                if len(data['V2w_y']) < 15:
                    os.remove(fpath)
                    continue
                V = header["Voltage_for_measurment"]
                Bext = header["Gausseter"] 
                if zerofield_correction and abs(Bext) < 10:  
                    zerofield_data[V] = data['V2w_y']
                    print('Zerofield data found for V =', V)
                else:
                    groups.setdefault(V, []).append(f)
                    

        outputs = {}

        color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        fig, ax = plt.subplots(2, 1, figsize=(10,14))

        for V, files in groups.items():
            if V == 1:
                continue
            print(f'Processing voltage: {V} V with files: {files}')

            # 2. Use .get() to safely pass None if V is not in zerofield_data
            zf = zerofield_data.get(V, None)
            outputs[V] = self.extract_AD_and_FL_and_thermal(files, plot=ax, V=V, color=next(color_cycle), zerofield_data=zf, plot_fits=plot_fits)
        ax[0].set_xlabel('1/(Bext + Bk) ($T^{-1}$)')
        ax[0].set_ylabel('Coefficient A = $\\left[ \\frac{R_{AHE}}{2} B_{AD} + T \\right]$(V)')
        ax[0].set_title('Field Dependence of A cos($\\phi$)')
        ax[0].legend()
        ax[1].set_xlabel('1/Bext  ($T^{-1}$)')
        ax[1].set_ylabel('Coefficient B = \\left[ 2 $R_{PHE}(B_{FL} + B_{Oe})$ (V) \\right]')
        ax[1].set_title('Field Dependence of $B (2cos^3(\\phi) - cos(\\phi))$ ')
        ax[1].legend()

        return outputs

    def plot_SOTs(self, results_dict):
        # 1. Extract and restructure the data
        voltages = []
        ad_sot = []
        ad_sot_err = []
        thermal = []
        thermal_err = []
        fl_sot = []
        fl_sot_err = []

        for voltage, inner_data in results_dict.items():
            
            voltages.append(voltage)
            ad_sot.append(float(inner_data['AD_SOT']))
            ad_sot_err.append(float(inner_data['AD_SOT_err']))
            thermal.append(float(inner_data['thermal']))
            thermal_err.append(float(inner_data['thermal_err']))
            fl_sot.append(float(inner_data['FL_SOT']))
            fl_sot_err.append(float(inner_data['FL_SOT_err']))

        # 2. Create a new DataFrame
        plot_df = pd.DataFrame({
            'Voltage (V)': voltages,
            'AD_SOT': ad_sot,
            "AD_SOT_err": ad_sot_err,
            'Thermal': thermal,
            "Thermal_err": thermal_err,
            'FL_SOT': fl_sot,
            "FL_SOT_err": fl_sot_err,
        })

        plot_df = plot_df.sort_values('Voltage (V)')
        x_data = plot_df['Voltage (V)'].values
        x_matrix = x_data[:, np.newaxis] # Needed for lstsq
        plt.figure(figsize=(10,6))
        ax = plt.gca() # Get current axis
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-8, -8)) # Force 10^-8
        ax.yaxis.set_major_formatter(formatter)

        # --- AD SOT Data and Fit ---
        plt.errorbar(plot_df['Voltage (V)'], plot_df['AD_SOT'], yerr=plot_df['AD_SOT_err'], 
                    marker='o', linestyle='none', label='AD SOT Data')

        # Fit and plot line through zero
        slope_ad, _, _, _ = np.linalg.lstsq(x_matrix, plot_df['AD_SOT'], rcond=None)
        x_fit = self.create_fit_array(np.min(plot_df['Voltage (V)']), np.max(plot_df['Voltage (V)']), num_points=200)
        plt.plot(x_fit, slope_ad * x_fit, '--', color='C0', alpha=0.7, 
                label=f'AD Fit (m={slope_ad[0]:.2e})')

        # --- FL SOT Data and Fit ---
        plt.errorbar(plot_df['Voltage (V)'], plot_df['FL_SOT'], yerr=plot_df['FL_SOT_err'], 
                    marker='o', linestyle='none', label='FL SOT Data')

        # Fit and plot line through zero
        slope_fl, _, _, _ = np.linalg.lstsq(x_matrix, plot_df['FL_SOT'], rcond=None)
        plt.plot(x_fit, slope_fl * x_fit, '--', color='C1', alpha=0.7, 
                label=f'FL Fit (m={slope_fl[0]:.2e})')

        # Formatting
        plt.xlabel('Applied Voltage (V)')
        plt.ylabel('Extracted Values')
        plt.title(f'Extracted SOTs vs Applied Voltage for {self.sample_name}')
        plt.axhline(0, color='black', linewidth=0.5) # Reference line at y=0
        plt.axvline(0, color='black', linewidth=0.5) # Reference line at x=0
        plt.legend()
        plt.show()
        print("Average AL SOT:", np.average(plot_df['AD_SOT']))
        print("Average FL SOT:", np.average(plot_df['FL_SOT']))


    def plot_folder(self, zerofield_correction=False):
        for f in os.listdir(self.folder):
            if f.lower().endswith(".txt"):
                fpath = os.path.join(self.folder, f)
                data, header = self.read_file(fpath)
                if len(data['V2w_y']) < 15:
                    os.remove(fpath)
                    continue
                self.fit_2nd_harm(data, header, plot=True)

    def create_fit_array(self, xmin, xmax, num_points=100):
        if xmin < 0:
            x_fit_min = xmin * 1.1
        if xmax < 0:
            x_fit_max = xmax * 0.9
        else:
            x_fit_min = xmin * 0.9
            x_fit_max = xmax * 1.1
        return np.linspace(x_fit_min, x_fit_max, num_points)

    def calculate_current_density(self, Vrms, R=50, rho_NM=10.6e-8, rho_FM=150e-8):
        # Calculate current amplitude from power in dBm
        Irms = Vrms/R  

        # Calculate current density (A/m^2), using geometry and material parameters
        width = self.Width * 1e-6  
        thickness_FM, thickness_NM = self.d_FM * 1e-9, self.d_NM * 1e-9  

        f_NM = (thickness_NM/rho_NM) / ((thickness_NM/rho_NM) + (thickness_FM/rho_FM))  
        I_NM = Irms * f_NM

        current_density = I_NM / (width * thickness_NM)  
        return current_density, I_NM, Irms





if __name__ == "__main__":
    folder = r".\\2ndHarmonicHall\\Data"
    file= r".\\2ndHarmonicHall\\Data\\FrequencyScan21_HI_3.00A_U_3.00V_HG_156.12mT.txt"

    hh = HH(folder=folder, file=file)
    data, header_data = hh.read_file(file)
    # hh.plot_first_harmonic(data, header_data)
    # hh.plot_both_harmonics(data, header_data)
    # hh.fit_2nd_harm(data, header_data)
    results = hh.extract_AD_and_FL_and_thermal(Bk=0.1)
    # print(results)
    # hh.initialize()



# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import re
import os 
import glob
import sys
from pathlib import Path
import itertools
import matplotlib.ticker as ticker
# STFMR Class 1.1
class HH:
    def __init__(self, folder=None, file=None, R=None, d_FM=None, d_NM=None, Ms=None, Width=None, Length=None, Bk=0, sample_name=None, zerofield_correction=True):
        if folder is not None:
            self.folder = folder
        if file is not None:
            self.file = file

        self.pattern = re.compile(
            r"HI_(?P<phi>[+-]?\d+(?:\.\d+)?)A_"
            r"U_(?P<freq>[+-]?\d+(?:\.\d+)?)V_"
            r"HG_(?P<pow>[+-]?\d+(?:\.\d+)?)mT"
        )

        # Physical constants
        self.mu_B = 9.274009994e-24  # Bohr magneton [J/T]
        self.e = 1.602176634e-19     # elementary charge [C]
        self.mu_0 = 4 * np.pi * 1e-7 # vacuum permeability [H/m]
        self.gamma = 1.760859e11     # gyromagnetic ratio [rad/(s·T)]
        self.hbar = 6.626e-34

        # System settings
        self.R = R
        self.d_FM = d_FM
        self.d_NM = d_NM
        self.Ms = Ms
        self.Width = Width
        self.Length = Length
        self.Bk = Bk
        self.sample_name = sample_name
        self.zerofield_correction = zerofield_correction

    # Helper functions
    def create_fit_array(self, xmin, xmax, num_points=100):
        if xmin < 0:
            x_fit_min = xmin * 1.1
        if xmax < 0:
            x_fit_max = xmax * 0.9
        else:
            x_fit_min = xmin * 0.9
            x_fit_max = xmax * 1.1
        return np.linspace(x_fit_min, x_fit_max, num_points)
    
    def get_key(self, filepath):
        """Create a key for a file, which describes the most important parameters of the datafile. Distinghuishes between measurement with or without DC."""
        filename = Path(filepath).name

        match = self.pattern.search(filename)
        if match:
            (DC, Voltage, Field) = match.groups()
            return (DC, Voltage, Field)
        return None
    
    def read_header(self, filepath):
        """
        Reads the first 5 header lines of the measurement file and stores
        the parameters as instance attributes.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            # Read exactly the first 5 parameter lines
            lines = [f.readline().strip() for _ in range(4)]
        
        header_data = {}
        # Pattern: "Label: number unit"
        pattern = r"^(.*?):\s*([+-]?\d*\.?\d+)\s*(.*)$"

        for line in lines:
            m = re.match(pattern, line)
            if m:
                label, value, unit = m.groups()
                key = label.replace(" ", "_")
                header_data[key] = float(value)
                header_data[key + "_unit"] = unit
        return header_data

    def read_file(self, filepath):
        """Reads a measurement file and returns the data as a pandas DataFrame."""
        data = pd.read_csv(filepath, skiprows=5, sep='\s+')
        header_data = self.read_header(filepath)

        # data['R1w'] = self.voltage_to_resistance(data['V1w'], self.R, header_data["Voltage_for_measurment"])
        data["R2w_x"] = self.voltage_to_resistance(data["V2w_x"], self.R, header_data["Voltage_for_measurment"])
        data["R2w_y"] = self.voltage_to_resistance(data["V2w_y"], self.R, header_data["Voltage_for_measurment"])
        return data, header_data

    def sort_files_by_voltage(self):
        # group by applied voltage
        groups = {}
        zerofield_data = {}

        for f in os.listdir(self.folder):
            if f.lower().endswith(".txt"):
                fpath = os.path.join(self.folder, f)
                data, header = self.read_file(fpath)
                if len(data['R2w_y']) < 15:
                    os.remove(fpath)
                    continue
                V = header["Voltage_for_measurment"]
                Bext = header["Gausseter"] 
                if abs(Bext) < 10:  
                    print('Zerofield data found for V =', V)
                    if self.zerofield_correction is True:
                        zerofield_data[V] = data['R2w_y']
                    elif self.zerofield_correction is None:
                        pass
                else:
                    groups.setdefault(V, []).append(f)
        return groups, zerofield_data

    # Quick plotting functions
    def plot_folder(self):
        for f in os.listdir(self.folder):
            if f.lower().endswith(".txt"):
                fpath = os.path.join(self.folder, f)
                data, header = self.read_file(fpath)
                if len(data['R2w_y']) < 15:
                    os.remove(fpath)
                    continue
                self.fit_2nd_harm(data, header, plot=True)

    def plot_first_harmonic(self, data, header_data):
        plt.figure(figsize=(10, 6))
        plt.plot(data['Angle'], data['V1w'], label='First Harmonic')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('First Harmonic Hall Voltage (V)')
        plt.title(f'First Harmonic vs Angle, B = {header_data["Gausseter"]} {header_data["Gausseter_unit"]} and V = {header_data["Voltage_for_measurment"]} {header_data["Voltage_for_measurment_unit"]}')
        plt.legend()
        plt.show()

    def plot_second_harmonic(self, data, header_data, textbox=True):
        plt.figure(figsize=(10, 6))
        plt.plot(data['Angle'], data['R2w_x']*1e6, label='Second Harmonic X-signal', color='orange')
        plt.plot(data['Angle'], data['R2w_y']*1e6, label='Second Harmonic Y-signal', color='green')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Second Harmonic Hall Voltage (uV)')
        plt.title(f'Second Harmonic vs Angle, B = {header_data["Gausseter"]} {header_data["Gausseter_unit"]} and V = {header_data["Voltage_for_measurment"]} {header_data["Voltage_for_measurment_unit"]}')
        plt.legend()
        plt.show()

    def plot_both_harmonics(self, data, header_data):
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Subplot: 1st harmonic
        axs[0].plot(data['Angle'], data['V1w'], label='First Harmonic')
        axs[0].set_ylabel('First Harmonic Hall Voltage (V)')
        axs[0].set_title(
            f'First Harmonic vs Angle, '
            f'B = {header_data["Gausseter"]} {header_data["Gausseter_unit"]}, '
            f'V = {header_data["Voltage_for_measurment"]} {header_data["Voltage_for_measurment_unit"]}'
        )        
        axs[0].legend()

        # Subplot: 2nd harmonic
        axs[1].plot(data['Angle'], data['R2w_x']*1e6,
                    label='Second Harmonic X-signal', color='orange')
        axs[1].plot(data['Angle'], data['R2w_y']*1e6,
                    label='Second Harmonic Y-signal', color='green')

        axs[1].set_xlabel('Angle (degrees)')
        axs[1].set_ylabel('Second Harmonic Hall Voltage (µV)')
        axs[1].set_title(
            f'Second Harmonic vs Angle, '
            f'B = {header_data["Gausseter"]} {header_data["Gausseter_unit"]}, '
            f'V = {header_data["Voltage_for_measurment"]} {header_data["Voltage_for_measurment_unit"]}'
        )
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    # Fitting with plotting functions
    def fit_2nd_harm(self, data, header_data, plot=False, cutoff=1):

        # Convert angles to radians for fitting
        angles_rad = np.deg2rad(data['Angle'])
        popt, pcov = curve_fit(self.angular_dependence,
                            angles_rad[cutoff:-cutoff],
                            data['R2w_y'][cutoff:-cutoff],)

        # Create phi in *degrees* but also phi_rad for evaluation
        phi = self.create_fit_array(np.min(data['Angle']), np.max(data['Angle']), num_points=200)
        phi_rad = np.deg2rad(phi)
        if plot:
            fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            # Plot raw data (degrees)
            axs[0].plot(data['Angle'], data['R2w_y'] * 1e6,
                        label='Second Harmonic Y-signal',
                        color='orange')

            # Plot fit (converted to radians)
            axs[0].plot(phi, self.angular_dependence(phi_rad, *popt) * 1e6,
                        label='Fit', color='red', linestyle='--')

            # Cosine component — also use radians!
            axs[0].plot(phi, popt[0] * np.cos(phi_rad + popt[3]) * 1e6 + popt[2] * 1e6,
                        label='Cosine Component', color='blue', linestyle=':')

            axs[0].set_ylabel('Second Harmonic Hall Voltage (V)')
            axs[0].set_title(
                f'Second Harmonic vs Angle, '
                f'B = {header_data["Gausseter"]} {header_data["Gausseter_unit"]}, '
                f'V = {header_data["Voltage_for_measurment"]} {header_data["Voltage_for_measurment_unit"]}'
            )        
            axs[0].legend()

            # ------------------------------
            # Subplot 2 — Second harmonic
            # ------------------------------
            dep = (data['R2w_y'] - popt[0]*np.cos(angles_rad))*1e6

            axs[1].plot(data['Angle'], dep,
                        label='Only ... depenence', color='orange')
            axs[1].plot(phi, (popt[1]*(2*np.cos(phi_rad+popt[3])**3 - np.cos(phi_rad+popt[3])) + popt[2])*1e6,
                        label='Fit', color='red', linestyle='--')
            axs[1].set_xlabel('Angle (degrees)')
            axs[1].set_ylabel('Second Harmonic Hall Voltage (µV)')
            axs[1].set_title(
                f'Second Harmonic vs Angle, '
                f'B = {header_data["Gausseter"]} {header_data["Gausseter_unit"]}, '
                f'V = {header_data["Voltage_for_measurment"]} {header_data["Voltage_for_measurment_unit"]}'
            )
            axs[1].legend()

            # Improve spacing
            plt.tight_layout()
            plt.show()
        return popt, pcov
    
    def plot_field_dependence(self):
        # Step 1: Group by voltage
        groups, zerofield_data = self.sort_files_by_voltage()

        # Step 2: For each voltage, fit all files and plot results
        for V, files in groups.items():
            print(f'Processing voltage: {V} V with files: {files}')
            file_field_pairs = []
            for f in files:
                _, header = self.read_file(os.path.join(self.folder, f))
                file_field_pairs.append((header["Gausseter"], f))
            
            # Sort the pairs by the field value (index 0)
            file_field_pairs.sort()
            
            # Extract the sorted filenames back out
            sorted_files = [pair[1] for pair in file_field_pairs]

            popts_list = []
            pcovs_list = []
            fields_FL = []
            fields_DL = []

            for file in sorted_files:
                data, header_data = self.read_file(os.path.join(self.folder, file))
                popt, pcov = self.fit_2nd_harm(data, header_data)
                popts_list.append(popt)  # mT → T
                pcovs_list.append(pcov)
                fields_FL.append(header_data["Gausseter"] / 1000)
                fields_DL.append(header_data["Gausseter"] / 1000 + self.Bk)
            pcovs_array = np.array(pcovs_list)      # shape: (N, n_params, n_params)
            yerr_A = np.sqrt(pcovs_array[:, 0, 0])             
            yerr_B = np.sqrt(pcovs_array[:, 1, 1])     

            fig, ax = plt.subplots(2, 1, figsize=(12,16))
            ax[0].errorbar(1/np.array(fields_DL), np.array(popts_list)[:, 0], yerr=yerr_A, marker='o', label=f'V = {V} V')
            ax[1].errorbar(1/np.array(fields_FL), np.array(popts_list)[:, 1], yerr=yerr_B, marker='o', label=f'V = {V} V')
            ax[0].set_xlabel('1/(Bext + Bk) ($T^{-1}$)')
            ax[0].set_ylabel('Coefficient A = $\\left[ \\frac{R_{AHE}}{2} B_{AD} + T \\right]$(V)')
            ax[0].set_title('Field Dependence of cos($\\phi$) ')
            ax[1].set_xlabel('1/Bext ($T^{-1}$)')
            ax[1].set_ylabel('Coefficient B = \\left[ 2 $R_{PHE}(B_{FL} + B_{Oe})$ (V) \\right]')
            ax[1].set_title('Field Dependence of ($2cos^3(\\phi) - cos(\\phi)$) ')
            ax[0].legend()
            ax[1].legend()
            plt.show()

    def extract_per_voltage(self, filelist, plot=True, V=None, color='blue', zerofield_data=False, plot_fits=False):
        results = []  
        # ---- STEP 1: Fit angular dependence for each field ----
        for file in filelist:
            filepath = os.path.join(self.folder, file)
            try:
                data, header = self.read_file(filepath)
                if abs(header["Gausseter"]) < 15:
                    continue
                if zerofield_data is True:
                    try:
                        data["R2w_y"] -= zerofield_data
                        # print(f'Applied zerofield correction to file {file}')
                    except ValueError as e:
                        print(f"Zero-field correction failed for {file}: {e}")


                # Fit model: A*cos(phi + phi0) + B*(2*cos^3(phi + phi0) - cos(phi + phi0)) + C

                popt, pcov = self.fit_2nd_harm(data=data, header_data=header, plot=plot_fits)
                A_fit, B_fit, C, phi0 = popt
                Bext = header["Gausseter"] / 1000  # mT → T
                
                # self.fit_2nd_harm(data, header, plot=True)
                results.append({
                    "file": file,
                    "Bext": Bext,
                    "A": A_fit,
                    "A_err": np.sqrt(pcov[0,0]),
                    "B": B_fit,
                    "B_err": np.sqrt(pcov[1,1]),
                    'Size': len(data['R2w_y'])
                })
            except:
                print(f'Failed to extract data from {file}, skipping file.')
                    
        # ---- STEP 2: AD + thermal extraction ----

        results = pd.DataFrame(results)
        fields_DL = results["Bext"] + self.Bk
        fields_FL = results["Bext"]
        S, S_err, C, C_err = self.fit_coeff_A(
            field=fields_DL,
            coeff_A=results["A"],
            sigma_coeff_A=results["A_err"]
        )

        # ---- STEP 3: FL extraction ----
        S_FL, S_FL_err, C_FL, C_FL_err = self.fit_coeff_B(
            field=fields_FL,
            coeff_B=results["B"],
            sigma_coeff_B=results["B_err"]
        )
        
        # Plot
        if plot is not None:
            fit_data_x = self.create_fit_array(min(1/fields_DL), max(1/fields_DL), num_points=100)
            fit_data_x_FL = self.create_fit_array(min(1/fields_FL), max(1/fields_FL), num_points=100)

            plot[0].errorbar(
                1/fields_DL, results["A"], yerr=results["A_err"],               
                ms=4,                    # Increase marker size (default is 6)
                fmt='o',
                markerfacecolor='black', 
                markeredgewidth=1,     # Thicker border around the point
                label=f'Data Points V = {V}', 
                color=color,
                elinewidth=1,            # Thicker error bar lines
                capsize=3,               # Width of the horizontal caps
            )            

            plot[1].errorbar(
                1/fields_FL, results["B"], yerr=results["B_err"],               
                ms=4,                    # Increase marker size (default is 6)
                fmt='o',
                markerfacecolor='black', 
                markeredgewidth=1,     # Thicker border around the point
                label=f'Data Points V = {V}', 
                color=color,
                elinewidth=1,            # Thicker error bar lines
                capsize=3,               # Width of the horizontal caps
            )       

            plot[0].plot(fit_data_x, S*fit_data_x + C, linestyle='--', color=color)
            # plot[0].set_xlim(left=-20)
            plot[1].plot(fit_data_x_FL, S_FL*fit_data_x_FL + C_FL, linestyle='--', color=color)
            # plot[1].set_xlim(left=-20)

        return {
            "results_per_field": results,
            "AD_SOT": S,
            "AD_SOT_err": S_err,
            "thermal": C,
            "thermal_err": C_err,
            "FL_SOT": S_FL,     # == 2Rphe x (Bfl + Boe) / Bext
            "FL_SOT_err": S_FL_err,    
        }

    def analyze_folder(self, plot_fits=False):
        groups, zerofield_data = self.sort_files_by_voltage()
                    
        self.outputs = {}

        color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        fig, ax = plt.subplots(2, 1, figsize=(10,14))

        for V, files in groups.items():
            print(f'Processing voltage: {V} V with files: {files}')

            # 2. Use .get() to safely pass None if V is not in zerofield_data
            zf = zerofield_data.get(V, None)
            self.outputs[V] = self.extract_per_voltage(files, plot=ax, V=V, color=next(color_cycle), zerofield_data=zf, plot_fits=plot_fits)
        ax[0].set_xlabel('1/(Bext + Bk) ($T^{-1}$)')
        ax[0].set_ylabel('Coefficient A = $\\left[ \\frac{R_{AHE}}{2} B_{AD} + T \\right]$(V)')
        ax[0].set_title('Field Dependence of A cos($\\phi$)')
        ax[0].legend()
        ax[1].set_xlabel('1/Bext  ($T^{-1}$)')
        ax[1].set_ylabel('Coefficient B = $\\left[ 2 R_{PHE}(B_{FL} + B_{Oe}) (V) \\right]$')
        ax[1].set_title('Field Dependence of $B (2cos^3(\\phi) - cos(\\phi))$ ')
        ax[1].legend()

        self.plot_SOTs()
        return 

    def plot_SOTs(self):
        if not hasattr(self, 'outputs'):
            print("No outputs found. Running extraction first.")
            self.analyze_folder()

        results_dict = self.outputs
        # 1. Extract and restructure the data
        voltages = []
        ad_sot = []
        ad_sot_err = []
        thermal = []
        thermal_err = []
        fl_sot = []
        fl_sot_err = []

        for voltage, inner_data in results_dict.items():
            
            voltages.append(voltage)
            ad_sot.append(float(inner_data['AD_SOT']))
            ad_sot_err.append(float(inner_data['AD_SOT_err']))
            thermal.append(float(inner_data['thermal']))
            thermal_err.append(float(inner_data['thermal_err']))
            fl_sot.append(float(inner_data['FL_SOT']))
            fl_sot_err.append(float(inner_data['FL_SOT_err']))

        # 2. Create a new DataFrame
        plot_df = pd.DataFrame({
            'Voltage (V)': voltages,
            'AD_SOT': ad_sot,
            "AD_SOT_err": ad_sot_err,
            'Thermal': thermal,
            "Thermal_err": thermal_err,
            'FL_SOT': fl_sot,
            "FL_SOT_err": fl_sot_err,
        })

        plot_df = plot_df.sort_values('Voltage (V)')
        x_data = plot_df['Voltage (V)'].values
        x_matrix = x_data[:, np.newaxis] # Needed for lstsq
        plt.figure(figsize=(10,6))
        ax = plt.gca() # Get current axis
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-8, -8)) # Force 10^-8
        ax.yaxis.set_major_formatter(formatter)

        # --- AD SOT Data and Fit ---
        plt.errorbar(plot_df['Voltage (V)'], plot_df['AD_SOT'], yerr=plot_df['AD_SOT_err'], 
                    marker='o', linestyle='none', label='AD SOT Data')

        # Fit and plot line through zero
        slope_ad, _, _, _ = np.linalg.lstsq(x_matrix, plot_df['AD_SOT'], rcond=None)
        x_fit = self.create_fit_array(0, np.max(plot_df['Voltage (V)']), num_points=200)
        plt.plot(x_fit, slope_ad * x_fit, '--', color='C0', alpha=0.7, 
                label=f'AD Fit (m={slope_ad[0]:.2e})')

        # --- FL SOT Data and Fit ---
        plt.errorbar(plot_df['Voltage (V)'], plot_df['FL_SOT'], yerr=plot_df['FL_SOT_err'], 
                    marker='o', linestyle='none', label='FL SOT Data')

        # Fit and plot line through zero
        slope_fl, _, _, _ = np.linalg.lstsq(x_matrix, plot_df['FL_SOT'], rcond=None)
        plt.plot(x_fit, slope_fl * x_fit, '--', color='C1', alpha=0.7, 
                label=f'FL Fit (m={slope_fl[0]:.2e})')

        # Formatting
        plt.xlabel('Applied Voltage (V)')
        plt.ylabel('Extracted Values')
        plt.title(f'Extracted SOTs vs Applied Voltage for {self.sample_name}')
        plt.axhline(0, color='black', linewidth=0.5) # Reference line at y=0
        plt.axvline(0, color='black', linewidth=0.5) # Reference line at x=0
        plt.legend()
        plt.show()
        print("Average AL SOT:", np.average(plot_df['AD_SOT']))
        print("Average FL SOT:", np.average(plot_df['FL_SOT']))

    # Equations and formulae
    def angular_dependence(self, phi, A, B, C, phi0):
        return A * np.cos(phi + phi0) + B * (2*np.cos(phi + phi0)**3 - np.cos(phi + phi0)) + C

    def fit_coeff_A(self, field, coeff_A, sigma_coeff_A):
        x = 1 / (field + self.Bk)
        y = coeff_A
        ysigma = sigma_coeff_A

        def linear_model(x, S, C):
            return S * x + C
        p, cov = curve_fit(linear_model, x, y, sigma=ysigma, absolute_sigma=True)
        slope, intercept = p                
        sigma_slope, sigma_intercept = np.sqrt(np.diag(cov))   
        return slope, sigma_slope, intercept, sigma_intercept
     
    def fit_coeff_B(self, field, coeff_B, sigma_coeff_B):
        x = 1 / (field)
        y = coeff_B
        ysigma = sigma_coeff_B

        def linear_model(x, S, C):
            return S * x + C
        p, cov = curve_fit(linear_model, x, y, sigma=ysigma, absolute_sigma=True)
        slope, intercept = p                   
        sigma_slope, sigma_intercept = np.sqrt(np.diag(cov))    
        return slope, sigma_slope, intercept, sigma_intercept

    def voltage_to_resistance(self, V_2w_xy, Vapp, R):
        Iac = Vapp / R
        R_2w_xy = V_2w_xy / Iac
        return R_2w_xy

    def calculate_current_density(self, Vrms, R=50, rho_NM=10.6e-8, rho_FM=150e-8):
        Irms = Vrms/R  

        # Calculate current density (A/m^2), using geometry and material parameters
        width = self.Width * 1e-6  
        thickness_FM, thickness_NM = self.d_FM * 1e-9, self.d_NM * 1e-9  

        f_NM = (thickness_NM/rho_NM) / ((thickness_NM/rho_NM) + (thickness_FM/rho_FM))  
        I_NM = Irms * f_NM

        current_density = I_NM / (width * thickness_NM)  
        return current_density, I_NM, Irms





if __name__ == "__main__":
    folder = r".\\2ndHarmonicHall\\Data"
    file= r".\\2ndHarmonicHall\\Data\\FrequencyScan21_HI_3.00A_U_3.00V_HG_156.12mT.txt"

    hh = HH(folder=folder, file=file)
    data, header_data = hh.read_file(file)
    # hh.plot_first_harmonic(data, header_data)
    # hh.plot_both_harmonics(data, header_data)
    # hh.fit_2nd_harm(data, header_data)
    # print(results)
    # hh.initialize()



