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
    def __init__(self, folder2nd=None, folder1st=None, file=None, R=None, Rxx=None, d_FM=None, d_NM=None, Ms=None, Width=None, Length=15e-9, Bk=0.8, sample_name=None, zerofield_correction=None, Rahe=None):
        if folder2nd is not None:
            self.folder2nd = folder2nd
        if folder1st is not None:
            self.folder1st = folder1st
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
        self.Rxx = Rxx
        self.Ms = Ms
        self.Bk = Bk
        self.Rahe = Rahe
        self.Rphe = None

        self.d_FM = d_FM
        self.d_NM = d_NM
        self.Width = Width
        self.Length = Length
        self.sample_name = sample_name
        self.zerofield_correction = zerofield_correction

        if self.Rxx is not None and self.Width is not None and self.Length is not None:
            self.resistivity = self.calculate_resistivity(self.Rxx, self.Width * (self.d_FM + self.d_NM), self.Length)  # in Ohm·m
            print("Calculated resistivity:", self.resistivity, "Ohm·m = ", self.resistivity*1e4, "µOhm·cm")   
        else:
            print("Warning: Rxx, Width, or Length not provided. Resistivity calculation skipped.")

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

        data['R1w'] = self.voltage_to_resistance(data['V1w'], header_data["Voltage_for_measurment"], self.R)
        data["R2w_x"] = self.voltage_to_resistance(data["V2w_x"], header_data["Voltage_for_measurment"], self.R)
        data["R2w_y"] = self.voltage_to_resistance(data["V2w_y"], header_data["Voltage_for_measurment"], self.R)
        return data, header_data

    def sort_files_by_voltage(self):
        groups = {}
        zerofield_data = {}

        for f in os.listdir(self.folder2nd):
            if not f.lower().endswith(".txt"):
                continue

            fpath = os.path.join(self.folder2nd, f)

            try:
                data, header = self.read_file(fpath)

                if len(data['R2w_y']) < 15:
                    os.remove(fpath)
                    continue

                V = round(float(header["Voltage_for_measurment"]), 3)
                Bext = float(header["Gausseter"])

                if abs(Bext) < 10:
                    print(f"Zerofield data found for V = {V}")
                    if self.zerofield_correction is True:
                        zerofield_data[V] = data['R2w_y']
                else:
                    groups.setdefault(V, []).append(f)

            except Exception as e:
                print(f"Failed to process {f}: {e}")

        return groups, zerofield_data

    # Quick plotting functions
    def plot_folder(self, fit=False):
        groups, zerofield_data = self.sort_files_by_voltage()

        # Step 2: For each voltage, fit all files and plot results
        for V, files in groups.items():
            for f in files:
                fpath = os.path.join(self.folder2nd, f)
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
    def fit_1st_harm(self, plot=False):
        Rphe_list = []
        Rphe_err_list = []
        for f in os.listdir(self.folder1st):
            if f.lower().endswith(".txt"):
                fpath = os.path.join(self.folder1st, f)
                
                data, header = self.read_file(fpath)
                if len(data['R2w_x']) < 15:
                    os.remove(fpath)
                    continue
                # Convert angles to radians for fitting
                angles_rad = np.deg2rad(data['Angle'])
                popt, pcov = curve_fit(self.angular_dependence_1st,
                                    angles_rad,
                                    data['R2w_x'], bounds=([0, -np.inf, -np.pi], [np.inf, np.inf, np.pi]))
                Rphe_list.append(popt[0])
                Rphe_err_list.append(np.sqrt(pcov[0,0]))
                # print(f'File: {f}, Extracted Rphe: {popt[0]} ± {np.sqrt(pcov[0,0])} Ohm')
                if plot:
                    # Create phi in *degrees* but also phi_rad for evaluation
                    phi = self.create_fit_array(np.min(data['Angle']), np.max(data['Angle']), num_points=200)
                    phi_rad = np.deg2rad(phi)

                    plt.figure(figsize=(10, 6))
                    # Plot raw data (degrees)
                    plt.plot(data['Angle'], data['R2w_x'],
                                label='First Harmonic Hall Voltage',
                                color='blue')

                    # Plot fit (converted to radians)
                    plt.plot(phi, self.angular_dependence_1st(phi_rad, *popt),
                                label='Fit', color='red', linestyle='--')

                    plt.xlabel('Angle (degrees)')
                    plt.ylabel('First Harmonic Hall Voltage (V)')
                    plt.title(
                        f'First Harmonic vs Angle, '
                        f'B = {header["Gausseter"]} {header["Gausseter_unit"]}, '
                        f'V = {header["Voltage_for_measurment"]} {header["Voltage_for_measurment_unit"]}')        
                    plt.legend()
                    plt.show()
        print('list', Rphe_list)
        self.Rphe = np.mean(Rphe_list)
        self.Rphe_err = np.sqrt(np.sum(np.array(Rphe_err_list)**2)) / len(Rphe_err_list)
        # print(Rphe_list, Rphe_err_list)
        print(f'Extracted Rphe: {self.Rphe} ± {self.Rphe_err} Ohm')
        return self.Rphe, self.Rphe_err
    
    # Fitting with plotting functions
    def fit_1st_harm_amr(self, plot=False):
        Ramr_list_R0 = []
        Ramr_err_list_R0 = []
        Ramr_list_DeltaR = []
        Ramr_err_list_DeltaR = []
        for f in os.listdir(self.folder1st):
            if f.lower().endswith(".txt"):
                fpath = os.path.join(self.folder1st, f)
                
                data, header = self.read_file(fpath)
                if len(data['R2w_x']) < 15:
                    os.remove(fpath)
                    continue
                # Convert angles to radians for fitting
                angles_rad = np.deg2rad(data['Angle'])
                popt, pcov = curve_fit(self.angular_dependence_amr,
                                    angles_rad,
                                    data['R2w_x'])
                Ramr_list_R0.append(popt[0])
                Ramr_err_list_R0.append(np.sqrt(pcov[0,0]))
                Ramr_list_DeltaR.append(popt[1])
                Ramr_err_list_DeltaR.append(np.sqrt(pcov[1,1]))
                # print(f'File: {f}, Extracted Rphe: {popt[0]} ± {np.sqrt(pcov[0,0])} Ohm')
                if plot:
                    # Create phi in *degrees* but also phi_rad for evaluation
                    phi = self.create_fit_array(np.min(data['Angle']), np.max(data['Angle']), num_points=200)
                    phi_rad = np.deg2rad(phi)

                    plt.figure(figsize=(10, 6))
                    # Plot raw data (degrees)
                    plt.plot(data['Angle'], data['R2w_x'],
                                label='First Harmonic Hall Voltage',
                                color='blue')

                    # Plot fit (converted to radians)
                    plt.plot(phi, self.angular_dependence_amr(phi_rad, *popt), label=f'R0={popt[0]:.2g}, DeltaR={popt[1]:.2g}', color='red', linestyle='--', )

                    plt.xlabel('Angle (degrees)')
                    plt.ylabel('First Harmonic Hall Voltage (V)')
                    plt.title(
                        f'First Harmonic vs Angle, '
                        f'B = {header["Gausseter"]} {header["Gausseter_unit"]}, '
                        f'V = {header["Voltage_for_measurment"]} {header["Voltage_for_measurment_unit"]}')        
                    plt.legend()
                    plt.show()
        print('list', Ramr_list_R0)
        self.R0 = np.mean(Ramr_list_R0)
        self.R0_err = np.sqrt(np.sum(np.array(Ramr_err_list_R0)**2)) / len(Ramr_err_list_R0)
        print('list', Ramr_list_DeltaR)
        self.DeltaR = np.mean(Ramr_list_DeltaR)
        self.DeltaR_err = np.sqrt(np.sum(np.array(Ramr_err_list_DeltaR)**2)) / len(Ramr_err_list_DeltaR)
        # print(Rphe_list, Rphe_err_list)
        print(f'Extracted R0: {self.R0} ± {self.R0_err} Ohm')
        print(f'Extracted DeltaR: {self.DeltaR} ± {self.DeltaR_err} Ohm')
        return self.R0, self.R0_err

    def fit_2nd_harm(self, data, header_data, plot=True, cutoff=1):
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
            axs[0].scatter(data['Angle'], data['R2w_y'] * 1e3,
                        label='Raw data',
                        color='orange')

            # Plot fit (converted to radians)
            axs[0].plot(phi, self.angular_dependence(phi_rad, *popt) * 1e3,
                        label='Fit', color='red', linestyle='--')

            # Cosine component — also use radians!
            axs[0].plot(phi, popt[0] * np.cos(phi_rad + popt[3]) * 1e3 + popt[2] * 1e3,
                        label='Cos($\\phi$) dependence', color='blue', linestyle=':')

            axs[0].set_ylabel('$R_{xy}^{2 \\omega}$ (m $\\Omega$)', fontsize=13)
            # axs[0].set_title(
            #     f'Second Harmonic vs Angle, '
            #     f'B = {header_data["Gausseter"]} {header_data["Gausseter_unit"]}, '
            #     f'V = {header_data["Voltage_for_measurment"]} {header_data["Voltage_for_measurment_unit"]}'
            # )        
            axs[0].legend()

            # ------------------------------
            # Subplot 2 — Second harmonic
            # ------------------------------
            dep = (data['R2w_y'] - popt[0]*np.cos(angles_rad))*1e3

            axs[1].scatter(data['Angle'], dep,
                        label='FL contribution', color='orange')
            axs[1].plot(phi, (popt[1]*(2*np.cos(phi_rad+popt[3])**3 - np.cos(phi_rad+popt[3])) + popt[2])*1e3,
                        label='Fit', color='red', linestyle='--')
            axs[1].set_xlabel('Angle (degrees)', fontsize=13)
            axs[1].set_ylabel('$R_{xy}^{2 \\omega}$ (m $\\Omega$)', fontsize=13)
            fig.suptitle(
                f"Second harmonic Hall resistance"
                # f"{header_data['Gausseter']} {header_data['Gausseter_unit']}, "
                # f"V = {header_data['Voltage_for_measurment']} {header_data['Voltage_for_measurment_unit']}"
            )
            axs[1].legend()
            print(f"{header_data['Gausseter']} {header_data['Gausseter_unit']}, ",

                f"V = {header_data['Voltage_for_measurment']} {header_data['Voltage_for_measurment_unit']}")
            # Improve spacing
            plt.tight_layout()
            plt.show()
        return popt, pcov
    

    def plot_field_dependence(self):
        # Step 1: Group by voltage
        groups, zerofield_data = self.sort_files_by_voltage()

        # Step 2: For each voltage, fit all files and plot results
        for V, files in sorted(groups.items()):            
            # print(f'Processing voltage: {V} V with files: {files}')
            file_field_pairs = []
            for f in files:
                _, header = self.read_file(os.path.join(self.folder2nd, f))
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
                data, header_data = self.read_file(os.path.join(self.folder2nd, file))
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
            ax[0].set_xlabel('1/(Bext + Bk) ($T^{-1}$)', fontsize=13)
            ax[0].set_ylabel('Coefficient A = $\\left[ \\frac{R_{AHE}}{2} B_{AD} + T \\right]$ ($\\Omega$)', fontsize=13)
            ax[0].set_title('Field Dependence of cos($\\phi$) ', fontsize=14)
            ax[1].set_xlabel('1/Bext ($T^{-1}$)',fontsize=13)
            ax[1].set_ylabel('Coefficient B = $\\left[ 2 R_{PHE}(B_{FL} + B_{Oe})$ ($\\Omega$) \\right]',fontsize=13)
            ax[1].set_title('Field Dependence of ($2cos^3(\\phi) - cos(\\phi)$) ', fontsize=14)
            ax[0].legend()
            ax[1].legend()
            plt.show()

    def extract_per_voltage(self, filelist, plot=True, V=None, color='blue', zerofield_data=None, plot_fits=True):
        results = []  
        # ---- STEP 1: Fit angular dependence for each field ----
        for file in filelist:
            filepath = os.path.join(self.folder2nd, file)
            try:
                data, header = self.read_file(filepath)
                if abs(header["Gausseter"]) < 15:
                    continue
                if zerofield_data is not None:
                    try:
                        data["R2w_y"] -= zerofield_data
                        # print(f'Applied zerofield correction to file {file}')
                        if V == 1:
                            print(zerofield_data[V])
                            print(data['R2w_y'])
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
                # print(f"Results {results} for {file}")
            except:
                print(f'Failed to extract data from {file}, skipping file.')
                    
        # ---- STEP 2: AD + thermal extraction ----
        results = pd.DataFrame(results)
        fields_DL = 1/ (results["Bext"] + self.Bk)
        fields_FL = 1/results["Bext"]
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
            fit_data_x = self.create_fit_array(min(fields_DL), max(fields_DL), num_points=100)
            fit_data_x_FL = self.create_fit_array(min(fields_FL), max(fields_FL), num_points=100)

            plot[0].errorbar(
                fields_DL, results["A"]*1e3, yerr=results["A_err"]*1e3,               
                ms=4,                    # Increase marker size (default is 6)
                fmt='o',
                markerfacecolor='black', 
                markeredgewidth=1,     # Thicker border around the point
                label=f'V$_{{app}}$ = {V}', 
                color=color,
                elinewidth=1,            # Thicker error bar lines
                capsize=3,               # Width of the horizontal caps
            )            

            plot[1].errorbar(
                fields_FL, results["B"]*1e3, yerr=results["B_err"]*1e3,               
                ms=4,                    # Increase marker size (default is 6)
                fmt='o',
                markerfacecolor='black', 
                markeredgewidth=1,     # Thicker border around the point
                label=f'V$_{{app}}$ = {V}', 
                color=color,
                elinewidth=1,            # Thicker error bar lines
                capsize=3,               # Width of the horizontal caps
            )       

            plot[0].plot(fit_data_x, (S*fit_data_x + C)*1e3, linestyle='--', color=color)
            # plot[0].set_xlim(left=-20)
            plot[1].plot(fit_data_x_FL, (S_FL*fit_data_x_FL + C_FL)*1e3, linestyle='--', color=color)
            # plot[1].set_xlim(left=-20)

        return {
            "results_per_field": results,
            "B_DL": S,
            "B_DL_err": S_err,
            "thermal": C,
            "thermal_err": C_err,
            "B_FL": S_FL,     # == 2Rphe x (Bfl + Boe) / Bext
            "B_FL_err": S_FL_err,    
        }

    def analyze_folder(self, plot_fits=False):
        groups, zerofield_data = self.sort_files_by_voltage()
                    
        self.outputs = {}

        color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        fig, ax = plt.subplots(2, 1, figsize=(10,14))

        for V, files in sorted(groups.items()):            
            if V < 1.5 or V > 4:
                continue
            print(f'Processing voltage: {V} V with files: {files}')

            # 2. Use .get() to safely pass None if V is not in zerofield_data
            zf = zerofield_data.get(V, None)
            self.outputs[V] = self.extract_per_voltage(files, plot=ax, V=V, color=next(color_cycle), zerofield_data=None, plot_fits=plot_fits)
        ax[0].set_xlabel('1/($B_{ext}$ + $B_{k}$) ($T^{-1}$)')
        ax[0].set_ylabel('Coefficient A = $\\left[R_{AHE} \, \\frac{B_{AD}}{B_{ext} + B_{k}} + T \\right] $ ($m \\Omega$)')
        ax[0].set_title('Field Dependence of A cos($\\phi$)')
        ax[0].legend()
        ax[1].set_xlabel('1/$B_{ext}$ ($T^{-1}$)')
        ax[1].set_ylabel('Coefficient B = $\\left[ 2 R_{PHE} \, \\frac{B_{FL} + B_{Oe}}{B_{ext}} \\right]$ ($m \\Omega$)')
        ax[1].set_title('Field Dependence of $B \, (2cos^3(\\phi) - cos(\\phi))$ ')
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
        B_DL = []
        B_DL_err = []
        thermal = []
        thermal_err = []
        B_FL = []
        B_FL_err = []

        for voltage, inner_data in results_dict.items():
            voltages.append(voltage)
            B_DL.append(float(inner_data['B_DL']))
            B_DL_err.append(float(inner_data['B_DL_err']))
            thermal.append(float(inner_data['thermal']))
            thermal_err.append(float(inner_data['thermal_err']))
            B_FL.append(float(inner_data['B_FL']))
            B_FL_err.append(float(inner_data['B_FL_err']))

        # 2. Create a new DataFrame and doing some calculations
        plot_df = pd.DataFrame({
            'Voltage (V)': voltages,
            'B_DL': B_DL,
            "B_DL_err": B_DL_err,
            'Thermal': thermal,
            "Thermal_err": thermal_err,
            'B_FL': B_FL,
            "B_FL_err": B_FL_err,
        })
        plot_df = plot_df.sort_values('Voltage (V)')
        plot_df['current_density'], plot_df['I_NM'] = self.calculate_current_density(plot_df["Voltage (V)"], self.R)[0], self.calculate_current_density(plot_df["Voltage (V)"], self.R)[1]  # Current in Amperes
        print(plot_df)

        if self.Rahe is None:
            self.Rahe = 1
        plot_df['B_DL'] = plot_df['B_DL'] / (self.Rahe)
        plot_df['B_DL_err'] = plot_df['B_DL_err'] / (self.Rahe)

        if self.Rphe is None:
            try:
                self.Rphe, self.Rphe_err = self.fit_1st_harm(plot=False)
            except:
                self.Rphe = 1e-2
        plot_df['B_FL'] = plot_df['B_FL'] / ( 2* self.Rphe)
        plot_df['B_FL_err'] = plot_df['B_FL_err'] / (2 * self.Rphe)

        # Calculate efficiencies using xi_ = 2e/hbar * (Ms * d_FM) * B_/J_NM
        # 3. Plotting
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))

        ax[0].errorbar(plot_df['current_density']/1e10, plot_df['B_DL']*1e3, yerr=plot_df['B_DL_err']*1e3, marker='o', linestyle='none', label='DL SOT')
        # ax[0].grid()

        ax[1].errorbar(plot_df['current_density']/1e10, plot_df['B_FL'], yerr=plot_df['B_FL_err'], marker='o', linestyle='none', label='FL SOT')
        # ax[2].errorbar(plot_df['current_density']/1e10, plot_df['Thermal'], yerr=plot_df['Thermal_err'], marker='o', linestyle='none', label='Thermal SOT Data')
        
        # 3. Fit lines through zero for both B_DL and B_FL
        def linear_model(x, S):
            return S * x 
        # print(plot_df[])
        # Fit and plot line through zero
        x_fit = self.create_fit_array(0, np.max(plot_df['current_density']), num_points=200)
        slope_ad, pcov_ad = curve_fit(linear_model, plot_df['current_density'], plot_df['B_DL'])

        # Slope and its error
        m_ad = slope_ad[0]
        sigma_m_ad = np.sqrt(pcov_ad[0, 0])

        # Constant prefactor
        C = (2 * self.e / self.hbar) * (self.Ms * self.d_FM)

        # DL efficiency and its absolute error
        DL_efficiency = m_ad * C
        DL_efficiency_err = sigma_m_ad * C
        slope_fl, _ = curve_fit(linear_model, plot_df['current_density'], plot_df['B_FL'])
        slope_thermal, _ = curve_fit(linear_model, plot_df['current_density'], plot_df['Thermal'])

        ax[0].plot(x_fit/1e10, slope_ad[0] * x_fit * 1e3, '--', color='C0', alpha=0.7, 
                label=f'DL Fit (slope={slope_ad[0]:.2e})')
        DL_efficiency = slope_ad[0] * (2 * self.e / self.hbar) * (self.Ms * self.d_FM)
        ax[1].plot(x_fit/1e10, slope_fl[0] * x_fit, '--', color='C1', alpha=0.7, 
                label=f'FL Fit (slope={slope_fl[0]:.2e})')
        FL_efficiency = slope_fl[0] * (2 * self.e / self.hbar) * (self.Ms * self.d_FM)
        # ax[2].plot(x_fit/1e10, slope_thermal[0] * x_fit, '--', color='C2', alpha=0.7, 
                # label=f'Thermal Fit (m={slope_thermal[0]:.2e})')

        # Formatting
        ax[0].set_xlabel('Charge current density in Pt ($\\times 10^{10} A/m^2$)', fontsize=14)
        ax[0].set_ylabel('B$_{DL}$ (mT)', fontsize=14)
        ax[1].set_xlabel('Charge current density in Pt ($\\times 10^{10} A/m^2$)', fontsize=14)
        ax[1].set_ylabel('B$_{FL}$ (mT)', fontsize=14)
        # ax[2].set_xlabel('Charge current density in Pt ($\\times 10^{10} A/m^2$)')
        # ax[2].set_ylabel('Thermal')
        plt.suptitle(f'Effective fields verus charge current density in Pt', fontsize=14)
        ax[0].legend(loc='upper left')
        ax[1].legend(loc='upper left')
        # plt.legend()
        plt.tight_layout()
        plt.show()
        print("DL_efficiency by fit with Jc", DL_efficiency, "±", DL_efficiency_err, "Where the slope is:", slope_ad)
        print("FL_efficiency by fit with Jc", FL_efficiency)

        # Calculate efficiencies using xi_ = 2e/hbar * (Ms * d_FM) * B_/E
        plot_df['E'] = self.calculate_E(self.resistivity, plot_df['current_density'])  # E in V/m
        # 3. Plotting
        fig, ax = plt.subplots(3, 1, figsize=(10, 14))

        ax[0].errorbar(plot_df['E'], plot_df['B_DL'], yerr=plot_df['B_DL_err'], marker='o', linestyle='none', label='DL SOT Data')
        ax[1].errorbar(plot_df['E'], plot_df['B_FL'], yerr=plot_df['B_FL_err'], marker='o', linestyle='none', label='FL SOT Data')
        ax[2].errorbar(plot_df['E'], plot_df['Thermal'], yerr=plot_df['Thermal_err'], marker='o', linestyle='none', label='Thermal SOT Data')
        
        # Fit and plot line through zero
        x_fit = self.create_fit_array(0, np.max(plot_df['E']), num_points=200)
        slope_ad, _ = curve_fit(linear_model, plot_df['E'], plot_df['B_DL'])
        slope_fl, _ = curve_fit(linear_model, plot_df['E'], plot_df['B_FL'])
        slope_thermal, _ = curve_fit(linear_model, plot_df['E'], plot_df['Thermal'])

        ax[0].plot(x_fit, slope_ad[0] * x_fit, '--', color='C0', alpha=0.7, 
                label=f'DL Fit (m={slope_ad[0]:.2e})')
        DL_efficiency = slope_ad[0] * (2 * self.e / self.hbar) * (self.Ms * self.d_FM)
        ax[1].plot(x_fit, slope_fl[0] * x_fit, '--', color='C1', alpha=0.7, 
                label=f'FL Fit (m={slope_fl[0]:.2e})')
        FL_efficiency = slope_fl[0] * (2 * self.e / self.hbar) * (self.Ms * self.d_FM)
        ax[2].plot(x_fit, slope_thermal[0] * x_fit, '--', color='C2', alpha=0.7, 
                label=f'Thermal Fit (m={slope_thermal[0]:.2e})')

        # Formatting
        ax[0].set_xlabel('Electric field in Pt (V/m)')
        ax[0].set_ylabel('B_DL (T)')
        ax[1].set_xlabel('Electric field in Pt (V/m)')
        ax[1].set_ylabel('B_FL (T)')
        ax[2].set_xlabel('Electric field in Pt (V/m)')
        ax[2].set_ylabel('Thermal')
        plt.suptitle(f'Extracted SOTs vs Charge Current for {self.sample_name}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        print("DL_efficiency by fit with E", DL_efficiency, "Using E=", self.calculate_E(self.resistivity, plot_df['current_density']), "Resistivity=", self.resistivity )
        print("FL_efficiency by fit with E", FL_efficiency)

    # Equations and formulae
    def angular_dependence(self, phi, A, B, C, phi0):
        return A * np.cos(phi + phi0) + B * (2*np.cos(phi + phi0)**3 - np.cos(phi + phi0)) + C

    def angular_dependence_1st(self, phi, Rphe, C, phi0):
        return Rphe * np.sin(2 * (phi + phi0)) + C
    
    def angular_dependence_amr(self, phi, R0, DeltaR, phi0):
        return R0 +  DeltaR * np.cos((phi + phi0))**2 #np.sin((phi + phi0)) * 
    
    def fit_coeff_A(self, field, coeff_A, sigma_coeff_A):
        x = field 
        y = coeff_A
        ysigma = sigma_coeff_A

        def linear_model(x, S, C):
            return S * x + C
        p, cov = curve_fit(linear_model, x, y, sigma=ysigma, absolute_sigma=True)
        slope, intercept = p                
        sigma_slope, sigma_intercept = np.sqrt(np.diag(cov))   
        return slope, sigma_slope, intercept, sigma_intercept

    def fit_coeff_B(self, field, coeff_B, sigma_coeff_B):
        x = field
        y = coeff_B
        ysigma = sigma_coeff_B

        def linear_model(x, S, C):
            return S * x + C
        p, cov = curve_fit(linear_model, x, y, sigma=ysigma, absolute_sigma=True)
        slope, intercept = p                   
        sigma_slope, sigma_intercept = np.sqrt(np.diag(cov))    
        return slope, sigma_slope, intercept, sigma_intercept

    def calculate_E(self, resistivity, Jc):
        E = resistivity * Jc
        return E

    def calculate_resistivity(self, R, A, L):
        rho = R * (A / L)
        return rho
    
    def calculate_DL_SOT_efficiency(self, B_DL, E):
        xi_DL = (2 * self.e / self.hbar) * (self.Ms * self.d_FM *  1e-9 * B_DL) / (E)
        return xi_DL

    def calculate_B_FL_efficiency(self, B_FL, E):
        xi_FL = (2 * self.e / self.hbar) * (self.Ms * self.d_FM *  1e-9 * B_FL) / (E)
        return xi_FL
    
    def voltage_to_resistance(self, V_2w_xy, Vapp, R):
        Iac = Vapp / R
        # print("Vapp:", Vapp, "R:", R, "Iac:", Iac)
        R_2w_xy = V_2w_xy / Iac
        return R_2w_xy

    def calculate_current_density(self, Vrms, R=50, rho_NM=10.6e-8, rho_FM=150e-8):
        Irms = Vrms/R  

        if self.d_FM > 1e-3:
            self.d_FM *= 1e-9
        if self.d_NM > 1e-3:
            self.d_NM *= 1e-9
        # Calculate current density (A/m^2), using geometry and material parameters
        if self.d_NM == 0:
            f_NM = 1
            self.d_NM = self.d_FM
        else:
            f_NM = (self.d_NM/rho_NM) / ((self.d_NM/rho_NM) + (self.d_FM/rho_FM))  
        I_NM = Irms * f_NM
        current_density = I_NM / (self.Width * self.d_NM)  
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



