
# ST-FMR analysis

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
from uncertainties import ufloat
from collections import defaultdict
import math

# STFMR Class 1.1
class STFMR:
    def __init__(self, folder=None, filename=None, thicknessFolder=None, voltage_unit='uV', d_FM=None, d_NM=None, Ms=None, Width=None, Length=None, rho_FM=None, rho_NM=None, used_fields='Pos', sample_name=None):
        """
        Parameters
        ----------
        filename : str
            Path to data file
        """
        if folder is not None:
            self.folder = folder
        if filename is not None:
            self.filename = filename
        if thicknessFolder is not None:
            self.thicknessFolder = thicknessFolder

        self.pattern = re.compile(
            r"Phi_(?P<phi>[+-]?\d+(?:\.\d+)?)deg_"
            r"Freq_(?P<freq>[+-]?\d+(?:\.\d+)?)GHz_"
            r"Pow_(?P<pow>[+-]?\d+(?:\.\d+)?)dBm_"
            r"Trace_(?P<trace>\d+)"
        )

        self.patternDC = re.compile(
            r"Phi_(?P<phi>[+-]?\d+(?:\.\d+)?)deg_"
            r"Freq_(?P<freq>[+-]?\d+(?:\.\d+)?)GHz_"
            r"Pow_(?P<pow>[+-]?\d+(?:\.\d+)?)dBm_"
            r"Curr_(?P<dc>[+-]?\d+(?:\.\d+)?)mA_"
            r"Trace_(?P<trace>\d+)"
        )

        # Physical constants
        self.mu_B = 9.274009994e-24  # Bohr magneton [J/T]
        self.e = 1.602176634e-19     # elementary charge [C]
        self.mu_0 = 4 * np.pi * 1e-7 # vacuum permeability [H/m]
        self.gamma = 1.760859e11     # gyromagnetic ratio [rad/(s·T)]
        self.hbar = 1.055e-34

        # System settings
        self.voltage_unit = voltage_unit  
        self.d_FM = d_FM
        self.rho_FM = rho_FM
        self.rho_NM = rho_NM
        self.d_NM = d_NM
        self.Ms = Ms
        self.Width = Width
        self.Length = Length
        self.used_fields=used_fields
        self.sample_name = sample_name
        if self.rho_FM is not None and self.rho_NM is not None:
            self.current_ditribution(self.rho_FM, self.rho_NM)

    def get_key(self, filepath):
        """Create a key for a file, which describes the most important parameters of the datafile. Distinghuishes between measurement with or without DC."""
        filename = Path(filepath).name
        if 'Curr' in filename:
            match = self.patternDC.search(filename)
            if match:
                phi, freq, pow_, dc, trace = match.groups()
                return (phi, freq, pow_, dc, trace)
            return None
        else:
            dc = 0
            match = self.pattern.search(filename)
            if match:
                phi, freq, pow_, trace = match.groups()
                return (phi, freq, pow_, dc, trace)
            return None
    
    def group_files(self, folder=None):
        """Groups files based on their key. Will be performed for the inputted folder in the class. """
        if folder is None:
            folder = self.folder

        grouped_files = {}
        for filename in os.listdir(folder):
            if filename.lower().endswith(".txt"): 
                (phi, freq, pow_, dc, trace) = self.get_key(filename)
                key = (phi, freq, pow_, dc)
                grouped_files.setdefault(key, []).append(filename)
            # print(grouped_files)
        return grouped_files

    def parse_resistance(self, first_line):
        # Look for "R = ... Ohm" with flexible spaces
        match = re.search(r"R\s*=\s*([0-9.]+)\s*Ohm", first_line)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"Could not parse resistance from line: '{first_line}'")
        
    def read_file(self, filepath):
         # Read the resistance value from the first line
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
        # Extract resistance value (assuming format 'R =93.6Ohm')
        try:
            R = self.parse_resistance(first_line)
        except ValueError as e:
            print(e)
            R = None  # or handle as needed
        # Read the actual data starting from the second line
        df = pd.read_csv(filepath, sep='\t', skiprows=1, header=0)
        average = ('H_interpolated' in df)

        if average:
            df = df.rename(columns={'H_interpolated': 'H_given'})
            df['H_mag'] = df['H_given']
        df = df.sort_values(by='H_given')

        # Make a copy of the magnitude column
        df["H_signed"] = df["H_mag"]
        H = df["H_given"].to_numpy()
        H_signed = H.copy()
        idx_half = len(H) // 2
        df = df.fillna(0)

        if min(H) >= 0 or average:
            return R, df
        else:
        # elif abs(H[0]) == abs(H[-1]):
            # print("Detected symmetric sweep: multiplying first half with -1.")
            df.loc[:idx_half, "H_signed"] *= -1
            if self.used_fields == 'Neg':
                # Keep only positive fields
                df = df.iloc[:idx_half+1].reset_index(drop=True)
            elif self.used_fields == 'Pos':
                # Keep only negative fields
                df = df.iloc[idx_half:].reset_index(drop=True)
            elif self.used_fields == 'Both':
                df.loc[:idx_half, "H_signed"] *= -1
            return R, df
    
    # -----------Analyzing and plotting data----------- 
    def current_ditribution(self, rho_FM, rho_NM, I0=None):
        if I0 is None:
            J_FM = 1/((self.Width*self.d_FM) * (1 + rho_FM/rho_NM))
            J_NM = 1/((self.Width*self.d_NM) * (1 + rho_FM/rho_NM))
            ratio = J_FM/J_NM
            print(f'Expected current distribution based on resistivities: FM: {ratio*100} % and NM: {(1/ratio)* 100} %')
        if I0 is not None:
            J_FM = I0/((self.Width*self.d_FM) * (1 + rho_FM/rho_NM))
            J_NM = I0/((self.Width*self.d_NM) * (1 + rho_FM/rho_NM))
            ratio = J_FM/J_NM
            print(f'Expected current distribution based on resistivities: FM: {ratio*100} % and NM: {(1/ratio)* 100} %')
            print(f'Currents: J_FM: {J_FM} % and J_NM: {(J_NM)} %')

    def fit_Vmix(self, filepath, plot=True, textbox=True, cutoff=3):
        # print(filepath)
        R, df = self.read_file(filepath)
        keys = self.get_key(filepath)

        # Perform fit using uV
        if self.voltage_unit == 'uV':
            df['Vx'] *= 1e6  # Convert Vx to microvolts
            df['Vy'] *= 1e6  # Convert Vy to microvolts
            df['Vx_err'] *= 1e6
            df['Vy_err'] *= 1e6
        elif self.voltage_unit == 'mV':
            df['Vx'] *= 1e3  # Convert Vx to millivolts
            df['Vy'] *= 1e3  # Convert Vy to millivolts
            df['Vx_err'] *= 1e3
            df['Vy_err'] *= 1e3

        if self.used_fields == 'Pos':
            H_min, H_max = 0, 400
        elif self.used_fields == 'Neg':
            H_min, H_max = -400, 0

        df_filtered = df[(df["H_signed"] >= H_min) & (df["H_signed"] <= H_max)] # creating a proper field range
        
        H = df_filtered["H_signed"][cutoff:-cutoff]
        V = df_filtered["Vx"][cutoff:-cutoff]
        sigma = df_filtered["Vx_err"][cutoff:-cutoff]

        # check if sigma is usable
        if sigma is not None and np.all(np.isfinite(sigma)) and not np.all(sigma == 0):
            mask = np.isfinite(H) & np.isfinite(V) & np.isfinite(sigma) & (sigma > 0)
            fit_params, cov = self.fit_stfmr(H[mask], V[mask], sigma=sigma[mask])
        else:
            mask = np.isfinite(H) & np.isfinite(V)
            fit_params, cov = self.fit_stfmr(H[mask], V[mask])
        if plot == True:
            ax = self.plot_Vmix(df_filtered, R, fit_params, keys, textbox=textbox)
            plt.show()
        return fit_params, cov

    def find_peak(self, H, V, cutoff=30):
        id_max = np.argmax(V[cutoff:-cutoff])
        id_min = np.argmin(V[cutoff:-cutoff])
        return np.average(id_max, id_min)

    def find_fit_range(self, H, V):
        len = len(H)
        step = H[1] - H[0]
        mid = self.find_peak(H, V)
        mid_to_max = H[-1] - H[mid]
        mid_to_min = H[mid] - H[min]
        max_steps = max(mid_to_max, mid_to_min)
        id_start = mid - max_steps
        id_stop = mid + max_steps
        return id_start, id_stop

    def plot_Vmix(self, df, R, fit_params=None, keys=None, textbox=False, ax=None, sample=None):
        if ax is None:
            plt.figure(figsize=(8,6))
            ax = plt.gca()
        ax.errorbar(df["H_given"], df["Vx"], df["Vx_err"], df.get("H_err"), fmt='.', label = sample if sample is not None else "data")

        H = self.create_fit_array(min(df["H_signed"]), max(df["H_signed"]), num_points=200)

        if fit_params is not None:      
            ax.plot(H, STFMR.stfmr_model(H, **fit_params), '-', label="fit")

            Vs, Va, Hres, Delta, C, D = fit_params["Vs"], fit_params["Va"], fit_params["Hres"], fit_params["Delta"], fit_params["C"], fit_params["D"]
            ax.plot(H, STFMR.stfmr_symmetric(H, Vs, Hres, Delta, C, D), '--', label="symmetric")
            ax.plot(H, STFMR.stfmr_asymmetric(H, Va, Hres, Delta, C, D), '--', label="antisymmetric")
        
        if keys is not None:
            # ax.set_title(f"Resistance={R}Ohm,Phi={keys[0]}deg, Freq={keys[1]}GHz, Pow={keys[2]}dBm, DC={keys[3]}mA, Trace={keys[4]}")
            ax.set_title(f"Mixing voltage versus magnetic field for frequency of {keys[1]} GHz") #{self.sample_name if self.sample_name is not None else ''}

        if textbox and fit_params is not None:
            # --- Add textbox with fit parameters ---
            position = 0.80 if self.used_fields == 'Pos' else 0.05
            textstr = '\n'.join((
                fr"$S = {Vs:.3g}\ \mu V$",
                fr"$A = {Va:.3g}\ \mu V$",
                fr"$H_{{res}} = {Hres:.3g}\ mT$",
                fr"$\Delta = {Delta:.3g}\ mT$",
                fr"$C = {C:.3g}\ \mu V$"
            ))
            # Place text box in lower right corner of the plot
            plt.gca().text(
                position, 0.22, textstr,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

        ax.set_xlabel("Magnetic Field (mT)")
        ax.set_ylabel(f"V$_{{mix}}$ ({self.voltage_unit})")       
        ax.legend()
        return ax
    
    def plot_general(self, df, col_x, col_y):
        """General plotting function to plot anything"""
        if col_x not in df.columns or col_y not in df.columns:
            raise ValueError(f"Columns '{col_x}' or '{col_y}' not found in DataFrame.")
    
        x = df[col_x]
        y = df[col_y]
        
        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(x, y)
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.title(f"Plot of {col_y} vs {col_x}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_traces(self):
        """Plotting different traces to check the differences between measurements"""
        if self.folder is not None:
            grouped_files = self.group_files()

            fig, ax = plt.subplots(len(grouped_files), 1, figsize=(8, 11*len(grouped_files)))
            for i, (key, files) in enumerate(grouped_files.items()):
                for f in files:
                    path = os.path.join(self.folder, f)
                    if 'Curr' in f:
                        (phi, freq, pow_, dc, trace) = self.get_key(f)
                        ax[i].set_title(f"Phi={phi}deg\nFreq={freq}GHz\nPow={pow_}dBm\nDC={dc}\nTrace={trace}")
                    else:
                        (phi, freq, pow_, dc, trace) = self.get_key(f)
                        ax[i].set_title(f"Phi={phi}deg\nFreq={freq}GHz\nPow={pow_}dBm\nTrace={trace}")
                    R, df = self.read_file(path)
                    # PLOT FUNCTION
                    ax[i].plot(df["H_signed"], df["Vx"], '.', label=f"Trace {trace}")
                    ax[i].set_xlabel("Field (mT)")
                    ax[i].set_ylabel(f"$Vmix$ ({self.voltage_unit})")
                    ax[i].legend(fontsize='small')
                    # PLOT FUNCTION
        elif self.filename is not None:
            pass
        else:
            sys.exit("Folder or filename not set. Please initialize STFMR with a folder path to datafiles.")

    def plot_quick_scans(self):
        """Plotting all traces of the folder in one figure"""
        plt.figure(figsize=(12,8))

        for file in os.listdir(self.folder):
            if not os.path.isfile(self.folder) and not file.lower().endswith(".txt"):
                    print(f"⚠️ Skipping {file}: no proper file.")
                    continue
            path = os.path.join(self.folder, file)
            key = self.get_key(file)
            R, df = self.read_file(path)
            plt.errorbar(df["H_given"], df["Vx"], df['Vx_err'], fmt='.', label=f"Frequency = {key[1]} GHz")
        plt.title(f"Scan of all frequencies, sample: {Path(self.folder).parts[0]}, resistance: {R:.2f} Ohm")
        plt.legend()
        plt.xlabel("Magnetic Field (mT)")
        plt.ylabel(f"Mixing Voltage ({self.voltage_unit})")

    def plot_averages(self):
        """Plotting the averages of all traces in one figure"""
        if self.used_fields== 'Pos':
            output_folder = os.path.join(self.folder, "Averaged_Traces_posB")
        elif self.used_fields== 'Neg':
            output_folder = os.path.join(self.folder, "Averaged_Traces_negB")
        
        if not os.path.exists(output_folder):
            self.analyze_traces()

        plt.figure(figsize=(8,6))
        for file in os.listdir(output_folder):
            if not os.path.isfile(file) and not file.lower().endswith(".txt"):
                    print(f"⚠️ Skipping {file}: no proper file.")
                    continue
            path = os.path.join(output_folder, file)
            key = self.get_key(file)
            R, df = self.read_file(path)
            avg_Vx = np.mean(df['Vx'])
            df['Vx'] *= 1e6
            df['Vx_err'] *= 1e6
            plt.errorbar(df["H_given"], df["Vx"] - avg_Vx*1e6, df['Vx_err'], fmt='.', label=f"Frequency = {key[1]} GHz") #power= {key[2]} dBm
            # Frequency = {key[1]} GHz,
        # plt.title(f"Scan of all frequencies, sample: {Path(self.folder).parts[0]}, resistance: {R:.2f} Ohm")
        plt.title(f"Scans for different frequencies")

        plt.legend()
        plt.xlabel("Magnetic Field (mT)")
        plt.ylabel(f"Mixing Voltage ({self.voltage_unit})")

    def analyze_traces(self, plot=False, folder=None):
        """Function that takes measurements from different traces. Subsequently averages all datapoints and performs a fit for these datapoints. """
        if folder is None:
            folder = self.folder

        try:
            if self.used_fields== 'Pos':
                output_folder = os.path.join(folder, "Averaged_Traces_posB")
            elif self.used_fields== 'Neg':
                output_folder = os.path.join(folder, "Averaged_Traces_negB")
        except AttributeError:
            sys.exit("Folder not set. Please initialize STFMR with a folder path to datafiles with multiple traces.")
        print(output_folder)
        # Create folder if not exists, and delete all files in them
        os.mkdir(output_folder) if not os.path.exists(output_folder) else None
        [os.remove(f.path) for f in os.scandir(output_folder) if f.is_file()]

        # Collect files by (Phi, Freq, Pow, DC)
        grouped_files = self.group_files(folder=folder)

        # Process each group
        for key, files in grouped_files.items():
            try:
                phi, freq, pow_, dc = key
                outname = f"Average_Phi_{phi}deg_Freq_{freq}GHz_Pow_{pow_}dBm_Curr_{dc}mA_Trace_0.txt"
                # print(f"Averaging files for key: {key} into {outname}")
            except:
                raise ValueError(f"Something went wrong, might be wrong key length: {len(key)}")
            dfs = []
            for f in files:
                path = os.path.join(folder, f)
                R, df = self.read_file(path)
                dfs.append(df)

            avg_df = self.average_traces(dfs)

            # Save averaged file
            outpath = os.path.join(output_folder, outname)
            with open(outpath, "w") as f:
                f.write(f"R ={R}Ohm\n")  # Write resistance
            avg_df.to_csv(outpath, sep="\t", index=False, mode="a")
            self.fit_Vmix(outpath, plot=plot)
        return 

    def average_traces(self, dfs, n_points=None):
        # 1. Determine overlapping H range        
        H_grid = dfs[0]["H_given"]
        
        # 3. Interpolate all runs onto H_grid
        H_all = []
        Vx_all = []
        Vy_all = []
        Vx_err_all = []
        Vy_err_all = []
        
        for df in dfs:
            df_sorted = df.sort_values("H_signed")

            # Linear interpolation for each column
            H_interp = np.interp(H_grid, df_sorted["H_signed"], df_sorted["H_signed"])
            Vx_interp = np.interp(H_grid, df_sorted["H_signed"], df_sorted["Vx"])
            Vy_interp = np.interp(H_grid, df_sorted["H_signed"], df_sorted["Vy"])
            
            # Handle errors: fill NaN with zero if needed
            Vx_err = df_sorted.get("Vx_err", pd.Series(0, index=df_sorted.index))
            Vy_err = df_sorted.get("Vy_err", pd.Series(0, index=df_sorted.index))
            Vx_err_interp = np.interp(H_grid, df_sorted["H_signed"], Vx_err.fillna(0))
            Vy_err_interp = np.interp(H_grid, df_sorted["H_signed"], Vy_err.fillna(0))

            H_all.append(H_interp)
            Vx_all.append(Vx_interp)
            Vy_all.append(Vy_interp)
            Vx_err_all.append(Vx_err_interp)
            Vy_err_all.append(Vy_err_interp)
        
        # 4. Convert to numpy arrays
        H_all = np.vstack(H_all)
        H_err = np.std(H_all, axis=0, ddof=1)
        Vx_all = np.vstack(Vx_all)
        Vy_all = np.vstack(Vy_all)
        Vx_err_all = np.vstack(Vx_err_all)
        Vy_err_all = np.vstack(Vy_err_all)
        
        # 5. Average values
        Vx_avg = np.mean(Vx_all, axis=0)
        Vy_avg = np.mean(Vy_all, axis=0)
        
        # 6. Compute propagated error (std across runs + measurement error)
        Vx_std = np.std(Vx_all, axis=0, ddof=1)
        Vy_std = np.std(Vy_all, axis=0, ddof=1)
        Vx_err_avg = np.sqrt(Vx_std**2 + np.mean(Vx_err_all**2, axis=0))
        Vy_err_avg = np.sqrt(Vy_std**2 + np.mean(Vy_err_all**2, axis=0))
        
        # 7. Build averaged DataFrame
        avg_df = pd.DataFrame({
            "H_interpolated": H_grid,
            "H_err": H_err,
            "Vx": Vx_avg,
            "Vy": Vy_avg,
            "Vx_err": Vx_err_avg,
            "Vy_err": Vy_err_avg
        })
        return avg_df


    def average_fitparams(self, plot=False, folder=None):
        """"Creates a .csv file with averages of the fitting variables from different measurements with different trace number.  """
        if folder is None:
            folder = self.folder
        
        try:
            if self.used_fields == 'Pos':
                resultpath = os.path.join(folder, "fit_results_average_posB.csv")
            elif self.used_fields == 'Neg': 
                resultpath = os.path.join(folder, "fit_results_average_negB.csv")
        except AttributeError:
            sys.exit("Folder not set. Please initialize STFMR with a folder path to datafiles.")
        print(resultpath)
        if os.path.exists(resultpath):
            os.remove(resultpath)  # delete existing results file, in case needs to be updated
            print(f"Deleted existing results file at {resultpath} to create a new one.")

        grouped_files = self.group_files(folder=folder)

        # Process each group   
        for key, files in grouped_files.items():
            temp = {"Vs": [], "Va": [], "Hres": [], "Delta": [], "C": [], "D": []}
            temp_cov = {"Vs": [], "Va": [], "Hres": [], "Delta": [], "C": [], "D": []}

            for f in files:
                print(f)
                fitparams_f, cov = self.fit_Vmix(os.path.join(folder, f), plot=plot)
                for i, name in enumerate(["Vs","Va","Hres","Delta","C", "D"]):
                    temp[name].append(fitparams_f[name])
                    temp_cov[name].append(np.sqrt(cov[i,i]))  # per-fit error

            fitparams_avg = {key: np.mean(value) for key, value in temp.items()}

            fitparams_std = {}
            for name in temp.keys():
                between_std = np.std(temp[name], ddof=1)
                within_std = np.sqrt(np.mean(np.array(temp_cov[name])**2))
                fitparams_std[name] = self.uncorrelated_errors(between_std, within_std)
            # else:
            #     fitparams_std = {k: np.std(v) for k, v in temp.items()}
            self.append_fit_results(f, fitparams_avg, std_dict=fitparams_std, folder=folder)
        return 
    
    def append_fit_results(self, filepath, fit_params, std_dict=None, folder=None):
        if folder is None:
            folder = self.folder

        filepath = Path(filepath)  # ensure Path
        has_curr = "Curr" in filepath.name

        # Extract keys
        phi, freq, pow_, dc, trace = self.get_key(filepath)
        # dc = None  # keep dc consistent

        # Decide filenames
        if self.used_fields == 'Pos':
            results_file = os.path.join(
                folder,
                "fit_results_average_posB.csv" if std_dict is not None else "fit_results_posB.csv"
            )
        elif self.used_fields == 'Neg': 
            results_file = os.path.join(
                folder,
                "fit_results_average_negB.csv" if std_dict is not None else "fit_results_negB.csv"
            )

        # Ensure file exists with header
        if not os.path.exists(results_file):
            with open(results_file, "w") as f:
                header = (
                        "Filename,phi,freq,pow_,dc,trace,"
                        "Vs,Vsstd,Va,Vastd,Hres,Hresstd,Delta,Deltastd,C,Cstd,D,Dstd\n"
                        if has_curr else
                        "Filename,phi,freq,pow_,trace,"
                        "Vs,Vsstd,Va,Vastd,Hres,Hresstd,Delta,Deltastd,C,Cstd,D,Dstd\n"
                    )
                f.write(header)
        # Append row
        with open(results_file, "a") as f:
            trace = "avg"
            row = (
                f"{filepath.name}, {phi}, {freq}, {pow_}, {dc}, {trace}, "
                f"{fit_params['Vs']},{std_dict['Vs']},{fit_params['Va']},{std_dict['Va']},"
                f"{fit_params['Hres']},{std_dict['Hres']},{fit_params['Delta']},{std_dict['Delta']},"
                f"{fit_params['C']},{std_dict['C']},{fit_params['D']},{std_dict['D']}\n"
                if has_curr else
                f"{filepath.name}, {phi}, {freq}, {pow_}, {trace}, "
                f"{fit_params['Vs']},{std_dict['Vs']},{fit_params['Va']},{std_dict['Va']},"
                f"{fit_params['Hres']},{std_dict['Hres']},{fit_params['Delta']},{std_dict['Delta']},"
                f"{fit_params['C']},{std_dict['C']},{fit_params['D']},{std_dict['D']}\n"
            )
            f.write(row)
    
    def angular_scan(self, method="Average_fitparams"):
        try:
            if self.used_fields == 'Pos':
                resultspath = os.path.join(self.folder, "fit_results_average_posB.csv")
            elif self.used_fields == 'Neg':
                resultspath = os.path.join(self.folder, "fit_results_average_negB.csv")
        except AttributeError:
            sys.exit("Folder not set. Please initialize STFMR with a folder path to datafiles.")
       
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False)


        df = pd.read_csv(resultspath)

        self.df = df.sort_values(by="phi")  # sort by angle

        f_S = lambda alpha, S_x_DL, S_y_DL, S_z_FL, phi0: (
        S_x_DL * np.sin(2*(alpha + phi0)) * np.sin(alpha + phi0) +
        S_y_DL * np.sin(2*(alpha + phi0)) * np.cos(alpha + phi0) +
        S_z_FL * np.sin(2*(alpha + phi0))
        )

        f_A = lambda alpha, A_x_FL, A_y_FL, A_z_DL, phi1: (
            A_y_FL * np.sin(2*(alpha + phi1)) * np.cos(alpha + phi1) +
            A_x_FL * np.sin(2*(alpha + phi1)) * np.sin(alpha + phi1) +
            A_z_DL * np.sin(2*(alpha + phi1))
        )

        # convert to radians for fitting
        phi_deg = self.df['phi'].to_numpy()
        phi_rad = np.mod(np.deg2rad(phi_deg), 2*np.pi)
        phi_deg_plot = self.create_fit_array(min(phi_deg), max(phi_deg), num_points=200)
        phi_plot_rad = np.mod(np.deg2rad(phi_deg_plot), 2*np.pi)
        print(min(df['Vs']))
        params_S, cov_S = curve_fit(f_S, phi_rad, self.df['Vs'], bounds=([-np.inf, -np.inf, -np.inf, -np.pi/4], [np.inf,np.inf, np.inf, np.pi/4]))
        params_A, cov_A = curve_fit(f_A, phi_rad, self.df['Va'], bounds=([-np.inf, -np.inf, -np.inf, -np.pi/4], [np.inf,np.inf, np.inf, np.pi/4]), maxfev=5000)

        # --- Plot Vs vs phi ---
        axes[0].errorbar(phi_deg, self.df["Vs"], yerr=self.df['Vsstd'], xerr=None, fmt="o",label="S")
        axes[0].plot(phi_deg_plot, f_S(phi_plot_rad, *params_S), label='Fit signal S', color='orange')
        
        S_x_DL, S_y_DL, S_z_FL, phi0 = params_S
        
        axes[0].plot(phi_deg_plot, S_x_DL * np.sin(2*(phi_plot_rad + phi0)) * np.sin(phi_plot_rad + phi0), label="$S^x_{{DL}}sin(2\\phi)sin(\\phi)$", linestyle='--')
        axes[0].plot(phi_deg_plot, S_y_DL * np.sin(2*(phi_plot_rad + phi0)) * np.cos(phi_plot_rad + phi0), label="$S^y_{{DL}}sin(2\\phi)cos(\\phi)$", linestyle='--')
        axes[0].plot(phi_deg_plot, S_z_FL * np.sin(2*(phi_plot_rad + phi0)), label="$S^z_{{FL}}sin(2\\phi)$", linestyle='--')
        axes[0].set_xlabel("phi (deg)")
        axes[0].set_ylabel(f"V ($\\mu$V)")
        # axes[0].set_title("A and S with outliers")
        # axes[0].set_ylim(bottom=-10, top=10)
        axes[0].grid(True)
        axes[0].legend(fontsize=14)

        # --- Plot Va vs phi ---
        axes[1].errorbar(phi_deg, self.df["Va"], yerr=self.df['Vastd'], xerr=None, fmt="s", color="red", label="A")
        axes[1].plot(phi_deg_plot, f_A(phi_plot_rad, *params_A), label='Fit signal A', color='orange')

        A_x_FL, A_y_FL, A_z_DL, phi1 = params_A
        axes[1].plot(phi_deg_plot, A_x_FL * np.sin(2*(phi_plot_rad + phi1)) * np.sin(phi_plot_rad + phi1), label=f"$A^x_{{FL}}sin(2\\phi)sin(\\phi)$", linestyle='--')
        axes[1].plot(phi_deg_plot, A_y_FL * np.sin(2*(phi_plot_rad + phi1)) * np.cos(phi_plot_rad + phi1), label=f"$A^y_{{FL}}sin(2\\phi)cos(\\phi)$", linestyle='--')
        axes[1].plot(phi_deg_plot, A_z_DL * np.sin(2*(phi_plot_rad + phi1)), label=f"$A^z_{{DL}}sin(2\\phi)$", linestyle='--')
        # axes[1].set_ylim(bottom=-10, top=10)

        axes[1].set_xlabel("phi (deg)")
        axes[1].set_ylabel(f"A ($\\mu$V)")
        # axes[1].set_title("Va and Vs versus phi")    
        axes[1].grid(True)
        axes[1].legend(fontsize=14)

        # # --- Plot Va vs phi ---
        # axes[2].plot(self.df["phi"], self.df["C"], "s-", color="orange", label="Vbias")
        # axes[2].set_xlabel("phi (deg)")
        # axes[2].set_ylabel(f"Vbias ({self.voltage_unit})")
        # axes[2].set_title("Vbias")
        # axes[2].grid(True)
        # axes[2].legend()

        # # # --- Plot Hres vs phi with filtering ---
        # avg_hres = self.df["Hres"].mean()
        # filtered_df = self.df[abs(self.df["Hres"] - avg_hres) <= 15].copy()
        # axes[3].errorbar(self.df["phi"], self.df["Hres"], yerr=self.df["Hresstd"], fmt="s-", color="orange", label="Hres (Filtered)")        
        # axes[3].set_xlabel("phi (deg)")
        # axes[3].set_ylabel(f"Hres (mT)")
        # axes[3].legend()

        plt.suptitle(f"Angular scan with frequency {self.df['freq'].iloc[1]} GHz for {self.sample_name}", fontsize=14)
        plt.tight_layout()
        plt.show()

        # plt.figure(figsize=(8,6))
        xi = self.FMR_efficiency(S_y_DL, A_x_FL, self.Ms, abs(self.d_FM), abs(self.d_NM), self.df['Hres'], 1)
        print(xi)
        # plt.plot(phi_deg, xi, 'o-')
        # plt.ylim(-0.5e-5, 0.5e-5)

        plt.show()

    def transverse_scan(self, method="Average_fitparams"):
        try:
            if self.used_fields == 'Pos':
                resultspath = os.path.join(self.folder, "fit_results_average_posB.csv")
            elif self.used_fields == 'Neg':
                resultspath = os.path.join(self.folder, "fit_results_average_negB.csv")
        except AttributeError:
            sys.exit("Folder not set. Please initialize STFMR with a folder path to datafiles.")

        fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

        df = pd.read_csv(resultspath)
        self.df = df.sort_values(by="phi")

        # --- Transverse fitting functions ---
        S_XY = lambda phi, S_XY_PHE_art, S_XY_AHE_art, S_XY_FL_PHE, S_XY_DL_AHE, phi0: (
            S_XY_PHE_art * np.cos(2*(phi - phi0)) * np.cos(phi - phi0) +
            S_XY_AHE_art * np.cos(phi - phi0) +
            S_XY_FL_PHE * np.cos(2*(phi - phi0)) +
            S_XY_DL_AHE
        )

        A_XY = lambda phi, A_XY_PHE, A_XY_AHE, A_XY_DL_PHE, A_XY_FL_AHE, phi0: (
            A_XY_PHE * np.cos(2*(phi - phi0)) * np.cos(phi - phi0) +
            A_XY_AHE * np.cos(phi - phi0) +
            A_XY_DL_PHE * np.cos(2*(phi - phi0)) +
            A_XY_FL_AHE
        )

        # --- Angle preparation ---
        phi_deg = self.df['phi'].to_numpy()
        phi_rad = np.mod(np.deg2rad(phi_deg), 2*np.pi)

        phi_deg_plot = self.create_fit_array(min(phi_deg), max(phi_deg), num_points=200)
        phi_plot_rad = np.mod(np.deg2rad(phi_deg_plot), 2*np.pi)

        # --- Fit ---
        params_S, cov_S = curve_fit(
            S_XY, phi_rad, self.df['Vs'],
            bounds=([-np.inf]*4 + [-np.pi/4], [np.inf]*4 + [np.pi/4])
        )

        params_A, cov_A = curve_fit(
            A_XY, phi_rad, self.df['Va'],
            bounds=([-np.inf]*4 + [-np.pi/4], [np.inf]*4 + [np.pi/4]),
            maxfev=5000
        )

        # =========================
        # --- Plot S (transverse) ---
        # =========================
        axes[0].errorbar(self.df["phi"], self.df["Vs"], yerr=self.df['Vsstd'],
                        fmt="o", label="$S_{XY}$", markersize=3)

        axes[0].plot(phi_deg_plot, S_XY(phi_plot_rad, *params_S),
                    label='Fit $S_{XY}$', color='orange')

        S_XY_PHE_art, S_XY_AHE_art, S_XY_FL_PHE, S_XY_DL_AHE, phi0 = params_S

        axes[0].plot(phi_deg_plot,
                    S_XY_PHE_art * np.cos(2*(phi_plot_rad - phi0)) * np.cos(phi_plot_rad - phi0),
                    linestyle='--', label=r"$S_{XY}^{PHE/art}$")

        axes[0].plot(phi_deg_plot,
                    S_XY_AHE_art * np.cos(phi_plot_rad - phi0),
                    linestyle='--', label=r"$S_{XY}^{AHE/art}$")

        axes[0].plot(phi_deg_plot,
                    S_XY_FL_PHE * np.cos(2*(phi_plot_rad - phi0)),
                    linestyle='--', label=r"$S_{XY}^{FL, PHE}$")

        axes[0].plot(phi_deg_plot,
                    np.full_like(phi_plot_rad, S_XY_DL_AHE),
                    linestyle='--', label=r"$S_{XY}^{DL, AHE}$")

        # axes[0].set_xlabel("phi (deg)", fontsize=12)
        axes[0].set_ylabel(r"S$_{XY}$ ($\mu$V)", fontsize=12)
        axes[0].grid(True)
        axes[0].legend(fontsize=12)

        # =========================
        # --- Plot A (transverse) ---
        # =========================
        axes[1].errorbar(self.df["phi"], self.df["Va"], yerr=self.df['Vastd'],
                        fmt="s", color="red", label="$A_{XY}$", markersize=3)

        axes[1].plot(phi_deg_plot, A_XY(phi_plot_rad, *params_A),
                    label='Fit $A_{XY}$', color='orange')

        A_XY_PHE, A_XY_AHE, A_XY_DL_PHE, A_XY_FL_AHE, phi1 = params_A

        axes[1].plot(phi_deg_plot,
                    A_XY_PHE * np.cos(2*(phi_plot_rad - phi1)) * np.cos(phi_plot_rad - phi1),
                    linestyle='--', label=r"$A_{XY}^{PHE}$")

        axes[1].plot(phi_deg_plot,
                    A_XY_AHE * np.cos(phi_plot_rad - phi1),
                    linestyle='--', label=r"$A_{XY}^{AHE}$")

        axes[1].plot(phi_deg_plot,
                    A_XY_DL_PHE * np.cos(2*(phi_plot_rad - phi1)),
                    linestyle='--', label=r"$A_{XY}^{DL,PHE}$")

        axes[1].plot(phi_deg_plot,
                    np.full_like(phi_plot_rad, A_XY_FL_AHE),
                    linestyle='--', label=r"$A_{XY}^{FL,AHE}$")

        axes[1].set_xlabel("phi (deg)", fontsize=12)
        axes[1].set_ylabel(r"A$_{XY}$ ($\mu$V)", fontsize=12)
        axes[1].grid(True)
        axes[1].legend(fontsize=12)

        # --- Title ---
        # plt.suptitle(
        #     f"Transverse angular scan with frequency {self.df['freq'].iloc[1]} GHz for {self.sample_name}",
        #     fontsize=14
        # )

        plt.tight_layout()
        plt.show()

    def angular_scan_transverse2(self, method='Average_fitparams'):
        """Analysis accoding based on paper ZrTe3 paper from Dan Ralph et al. """
        sample_folder = self.folder

        transverse_folder = os.path.join(sample_folder, "Angular_transverse")
        longitudinal_folder = os.path.join(sample_folder, "Angular_longitudinal")

        resultspath_transverse = os.path.join(transverse_folder, "fit_results_average_posB.csv")
        resultspath_longitudinal = os.path.join(longitudinal_folder, "fit_results_average_posB.csv")

        resultspath_transverse_avg = os.path.join(transverse_folder, "Averaged_Traces_posB", "fit_results_average_posB.csv")
        resultspath_longitudinal_avg = os.path.join(longitudinal_folder, "Averaged_Traces_posB", "fit_results_average_posB.csv")

        if not os.path.exists(resultspath_transverse):
            self.average_fitparams(folder=transverse_folder, plot=False)
        if not os.path.exists(resultspath_transverse_avg):
            self.analyze_traces(folder=transverse_folder)
            self.average_fitparams(plot=False, folder=Path(resultspath_transverse_avg).parent)

        if not os.path.exists(resultspath_longitudinal):
            self.average_fitparams(folder=longitudinal_folder)
        if not os.path.exists(resultspath_longitudinal_avg):
            self.analyze_traces(folder=longitudinal_folder)
            self.average_fitparams(plot=False, folder=Path(resultspath_longitudinal_avg).parent)

        if method == 'Average_fitparams':
            df_transverse = pd.read_csv(resultspath_transverse)
            df_longitudinal = pd.read_csv(resultspath_longitudinal)
        if method == 'Average_measurements':
            df_transverse = pd.read_csv(resultspath_transverse_avg)
            df_longitudinal = pd.read_csv(resultspath_longitudinal_avg)

        df_transverse = df_transverse.sort_values(by="phi").reset_index(drop=True)  # sort by angle
        df_longitudinal = df_longitudinal.sort_values(by="phi").reset_index(drop=True)  # sort by angle

        # convert to radians for fitting
        phi_deg_transverse = df_transverse['phi'].to_numpy()
        phi_deg_longitudinal = df_longitudinal['phi'].to_numpy()

        phi_rad_transverse = np.mod(np.deg2rad(phi_deg_transverse), 2*np.pi)
        phi_rad_longitudinal = np.mod(np.deg2rad(phi_deg_longitudinal), 2*np.pi)

        phi_deg_plot = self.create_fit_array(min(phi_deg_transverse), max(phi_deg_transverse), num_points=200)
        phi_plot_rad = np.mod(np.deg2rad(phi_deg_plot), 2*np.pi)

        S_XX = lambda phi, S_XX_AMR_art, S_XX_FL, phi0: (
        S_XX_AMR_art * np.sin(2*(phi-phi0)) * np.cos(phi-phi0) +
        S_XX_FL * np.sin(2*(phi-phi0))  
        )

        A_XX = lambda phi, A_XX_AMR, A_XX_DL, phi0: (
        A_XX_AMR * np.sin(2*(phi-phi0)) * np.cos(phi-phi0) +
        A_XX_DL * np.sin(2*(phi-phi0))  
        )

        S_XY = lambda phi, S_XY_PHE_art, S_XY_AHE_art, S_XY_FL_PHE, S_XY_DL_AHE, phi0: (
        S_XY_PHE_art * np.cos(2*(phi-phi0)) * np.cos(phi-phi0) +
        S_XY_AHE_art * np.cos(phi-phi0) +
        S_XY_FL_PHE * np.cos(2*(phi-phi0)) +
        S_XY_DL_AHE
        )

        A_XY = lambda phi, A_XY_PHE, A_XY_AHE, A_XY_DL_PHE, A_XY_FL_AHE, phi0: (
        A_XY_PHE * np.cos(2*(phi-phi0)) * np.cos(phi-phi0) +
        A_XY_AHE * np.cos(phi-phi0) +
        A_XY_DL_PHE * np.cos(2*(phi-phi0)) +
        A_XY_FL_AHE 
        )
        
        params_Vs_XX, cov_Vs_XX = curve_fit(S_XX, phi_rad_longitudinal, df_longitudinal['Vs'], bounds=([1.5*min(df_longitudinal['Vs']), 1.5*min(df_longitudinal['Vs']), -np.pi/4], [1.5*max(df_longitudinal['Vs']), 1.5*max(df_longitudinal['Vs']), np.pi/4]))
        params_Va_XX, cov_Va_XX = curve_fit(A_XX, phi_rad_longitudinal, df_longitudinal['Va'], bounds=([1.5*min(df_longitudinal['Va']), 1.5*min(df_longitudinal['Va']), -np.pi/4], [1.5*max(df_longitudinal['Va']), 1.5*max(df_longitudinal['Va']), np.pi/4]))
        S_XX_AMR_art, S_XX_FL, phi0_XX_ = params_Vs_XX
        A_XX_AMR, A_XX_DL, phi0_XX = params_Va_XX
        S_XX_AMR_art_err, S_XX_FL_err, phi0_XX_err = np.sqrt(np.diag(cov_Vs_XX))
        A_XX_AMR_err, A_XX_DL_err, phi0_XX_err = np.sqrt(np.diag(cov_Va_XX))
        

        params_Vs_XY, cov_Vs_XY = curve_fit(S_XY, phi_rad_transverse, df_transverse['Vs'], bounds=([1.5*min(df_transverse['Vs']), 1.5*min(df_transverse['Vs']), 1.5*min(df_transverse['Vs']), 1.5*min(df_transverse['Vs']), -np.pi/4], [1.5*max(df_transverse['Vs']),1.5*max(df_transverse['Vs']), 1.5*max(df_transverse['Vs']), 1.5*max(df_transverse['Vs']), np.pi/4]))
        params_Va_XY, cov_Va_XY = curve_fit(A_XY, phi_rad_transverse, df_transverse['Va'], p0=[2, 1, 0, 0, 0], bounds=([1.5*min(df_transverse['Va']), 1.5*min(df_transverse['Va']), 1.5*min(df_transverse['Va']), 1.5*min(df_transverse['Va']), -np.pi/4], [1.5*max(df_transverse['Va']), 1.5*max(df_transverse['Va']), 1.5*max(df_transverse['Va']), 1.5*max(df_transverse['Va']), np.pi/4]))
        S_XY_PHE_art, S_XY_AHE_art, S_XY_FL_PHE, S_XY_DL_AHE, phi0_XY_ = params_Vs_XY*-1
        A_XY_PHE, A_XY_AHE, A_XY_DL_PHE, A_XY_FL_AHE, phi0_XY = params_Va_XY*-1
        S_XY_PHE_art_err, S_XY_AHE_art_err, S_XY_FL_PHE_err, S_XY_DL_AHE_err, phi0_XY_err = np.sqrt(np.diag(cov_Vs_XY))
        A_XY_PHE_err, A_XY_AHE_err, A_XY_DL_PHE_err, A_XY_FL_AHE_err, phi0_XY_err = np.sqrt(np.diag(cov_Va_XY))

        print("---- Angular fitting results ----\n")
        print("Longitudinal mixing voltage:")
        print(f"  Vs (symmetric):")
        print(f"    S_XX_AMR_art = {S_XX_AMR_art:.4g} ± {S_XX_AMR_art_err:.2g}")
        print(f"    S_XX_FL      = {S_XX_FL:.4g} ± {S_XX_FL_err:.2g}")

        print(f"  Va (antisymmetric):")
        print(f"    A_XX_AMR     = {A_XX_AMR:.4g} ± {A_XX_AMR_err:.2g}")
        print(f"    A_XX_DL      = {A_XX_DL:.4g} ± {A_XX_DL_err:.2g}")

        print("Transverse mixing voltage:")
        print(f"  Vs (symmetric):")
        print(f"    S_XY_PHE_art  = {S_XY_PHE_art:.4g} ± {S_XY_PHE_art_err:.2g}")
        print(f"    S_XY_AHE_art  = {S_XY_AHE_art:.4g} ± {S_XY_AHE_art_err:.2g}")
        print(f"    S_XY_FL_PHE   = {S_XY_FL_PHE:.4g} ± {S_XY_FL_PHE_err:.2g}")
        print(f"    S_XY_DL_AHE   = {S_XY_DL_AHE:.4g} ± {S_XY_DL_AHE_err:.2g}")

        print(f"  Va (antisymmetric):")
        print(f"    A_XY_PHE      = {A_XY_PHE:.4g} ± {A_XY_PHE_err:.2g}")
        print(f"    A_XY_AHE      = {A_XY_AHE:.4g} ± {A_XY_AHE_err:.2g}")
        print(f"    A_XY_DL_PHE      = {A_XY_DL_PHE:.4g} ± {A_XY_DL_PHE_err:.2g}")
        print(f"    A_XY_FL_AHE      = {A_XY_FL_AHE:.4g} ± {A_XY_FL_AHE_err:.2g}")

        # Longitudinal

        Vs_XX_components = [
            {"func": lambda phi: S_XX_AMR_art * np.sin(2*(phi-phi0_XX_)) * np.cos(phi-phi0_XX_),  "label": "$S_{XX}^{AMR/art}$"},
            {"func": lambda phi: S_XX_FL * np.sin(2*(phi-phi0_XX))  ,           "label": "$S_{XX}^{FL}$"},
        ]
        Va_XX_components = [
            {"func": lambda phi: A_XX_AMR * np.sin(2*(phi-phi0_XX)) * np.cos(phi-phi0_XX),  "label": "$A_{XX}^{AMR}$"},
            {"func": lambda phi: A_XX_DL * np.sin(2*(phi-phi0_XX)),       "label": "$A_{XX}^{DL}$"},
        ]

        self._plot_angular_scan(
            df=df_longitudinal,
            phi_deg_plot=phi_deg_plot,
            phi_plot_rad=phi_plot_rad,
            Vs_func=S_XX,
            Va_func=A_XX,
            params_Vs=params_Vs_XX,
            params_Va=params_Va_XX,
            label=f"Longitudinal scan sample {os.path.basename(sample_folder)}",
            Vs_components=Vs_XX_components,
            Va_components=Va_XX_components,
        )

        Vs_XY_components = [
            {"func": lambda phi: S_XY_PHE_art * np.cos(2*(phi-phi0_XY_)) * np.cos(phi-phi0_XY_), "label": "$S_{XY}^{PHE/art}$"},
            {"func": lambda phi: S_XY_AHE_art * np.cos(phi-phi0_XY_),                             "label": "$S_{XY}^{AHE/art}$"},
            {"func": lambda phi: S_XY_FL_PHE  * np.cos(2*(phi-phi0_XY_)),                         "label": "$S_{XY}^{FL/PHE}$"},
            {"func": lambda phi: np.full_like(phi, S_XY_DL_AHE),                                  "label": "$S_{XY}^{DL/AHE}$"},
        ]
        Va_XY_components = [
            {"func": lambda phi: A_XY_PHE     * np.cos(2*(phi-phi0_XY)) * np.cos(phi-phi0_XY), "label": "$A_{XY}^{PHE}$"},
            {"func": lambda phi: A_XY_AHE     * np.cos(phi-phi0_XY),                            "label": "$A_{XY}^{AHE}$"},
            {"func": lambda phi: A_XY_DL_PHE  * np.cos(2*(phi-phi0_XY)),                        "label": "$A_{XY}^{DL/PHE}$"},
            {"func": lambda phi: np.full_like(phi, A_XY_FL_AHE),                                "label": "$A_{XY}^{FL/AHE}$"},
        ]

        # Transverse
        self._plot_angular_scan(
            df=df_transverse,
            phi_deg_plot=phi_deg_plot,
            phi_plot_rad=phi_plot_rad,
            Vs_func=S_XY,
            Va_func=A_XY,
            params_Vs=params_Vs_XY,
            params_Va=params_Va_XY,
            label=f"Transverse scan sample {os.path.basename(sample_folder)}",
            Vs_components=Vs_XY_components,
            Va_components=Va_XY_components,
        )


        def solve_E_art_1(A_XY_AHE, S_XY_AHE_art, S_XY_PHE_art, A_XY_PHE,
            sigma_A_XY_AHE, sigma_S_XY_AHE_art, sigma_S_XY_PHE_art, sigma_A_XY_PHE,
        ):
            S1 = S_XY_AHE_art
            S2 = S_XY_PHE_art
            A1 = A_XY_AHE
            A2 = A_XY_PHE
            W = self.Width
            sigma_Width = 0

            D = (S1 - S2)**2 - 4 * A1 * A2
            if D < 0:
                raise ValueError("Discriminant is negative; no real solutions.")

            sqrtD = math.sqrt(D)

            E_plus  = (-(S1 + S2) + sqrtD) / W
            E_minus = (-(S1 + S2) - sqrtD) / W

            def propagated_sigma(sign):
                # sign = +1 for E_plus, -1 for E_minus
                dE_dA1 = sign * (-2 * A2) / (W * sqrtD)
                dE_dA2 = sign * (-2 * A1) / (W * sqrtD)
                dE_dS1 = (-1 + sign * (S1 - S2) / sqrtD) / W
                dE_dS2 = (-1 - sign * (S1 - S2) / sqrtD) / W

                E = E_plus if sign == 1 else E_minus
                dE_dW = -E / W if sigma_Width > 0 else 0.0

                var = (
                    (dE_dA1 * sigma_A_XY_AHE)**2 +
                    (dE_dA2 * sigma_A_XY_PHE)**2 +
                    (dE_dS1 * sigma_S_XY_AHE_art)**2 +
                    (dE_dS2 * sigma_S_XY_PHE_art)**2 +
                    (dE_dW  * sigma_Width)**2
                )
                return math.sqrt(var)

            sigma_E_plus = propagated_sigma(+1)
            sigma_E_minus = propagated_sigma(-1)

            return E_plus, sigma_E_plus, E_minus, sigma_E_minus

        def solve_E_art_2(
            A_XY_AHE, S_XY_AHE_art, S_XX_AMR_art, A_XX_AMR,
            sigma_A_XY_AHE, sigma_S_XY_AHE_art, sigma_S_XX_AMR_art, sigma_A_XX_AMR,
        ):
            A1 = A_XY_AHE
            S1 = S_XY_AHE_art
            S2 = S_XX_AMR_art
            A2 = A_XX_AMR
            W, sigma_Width = self.Width, 0
            L, sigma_Length = self.Length, 0
            
            D = (S1 * L - S2 * W)**2 - 4 * W * L * A1 * A2
            if D < 0:
                raise ValueError("Discriminant is negative; no real solutions.")

            sqrtD = math.sqrt(D)

            E_plus  = (-(S1 * L + S2 * W) + sqrtD) / (W * L)
            E_minus = (-(S1 * L + S2 * W) - sqrtD) / (W * L)

            def sigma_E(sign):
                # sign = +1 for E_plus, -1 for E_minus

                dE_dA1 = sign * (-2 * A2) / sqrtD
                dE_dA2 = sign * (-2 * A1) / sqrtD

                term = (S1 * L - S2 * W) / sqrtD
                dE_dS1 = (-1 + sign * term) / W
                dE_dS2 = (-1 - sign * term) / L

                var = (
                    (dE_dA1 * sigma_A_XY_AHE)**2 +
                    (dE_dA2 * sigma_A_XX_AMR)**2 +
                    (dE_dS1 * sigma_S_XY_AHE_art)**2 +
                    (dE_dS2 * sigma_S_XX_AMR_art)**2
                )

                # Optional: include Width and Length uncertainties numerically
                if sigma_Width > 0 or sigma_Length > 0:
                    def E_of(Wloc, Lloc):
                        Dloc = (S1 * Lloc - S2 * Wloc)**2 - 4 * Wloc * Lloc * A1 * A2
                        if Dloc < 0:
                            return float("nan")
                        return (-(S1 * Lloc + S2 * Wloc) + sign * math.sqrt(Dloc)) / (Wloc * Lloc)

                    # finite differences
                    if sigma_Width > 0:
                        hW = max(1e-12, abs(W) * 1e-8)
                        dE_dW = (E_of(W + hW, L) - E_of(W - hW, L)) / (2 * hW)
                        var += (dE_dW * sigma_Width) ** 2

                    if sigma_Length > 0:
                        hL = max(1e-12, abs(L) * 1e-8)
                        dE_dL = (E_of(W, L + hL) - E_of(W, L - hL)) / (2 * hL)
                        var += (dE_dL * sigma_Length) ** 2

                return math.sqrt(var)

            sigma_E_plus = sigma_E(+1)
            sigma_E_minus = sigma_E(-1)

            return E_plus, sigma_E_plus, E_minus, sigma_E_minus


        def eta1(A_XY_AHE, S_XY_AHE_art, E_art,
                sigma_A_XY_AHE, sigma_S_XY_AHE_art, sigma_E_art):

            A = A_XY_AHE
            S = S_XY_AHE_art
            E = E_art
            W, sigma_Width = self.Width, 0

            denom = S + W * (E / 2)

            if denom == 0:
                raise ValueError("Denominator is zero; eta1 is undefined.")

            eta = -A / denom

            d_eta_dA = -1 / denom
            d_eta_dS = A / (denom ** 2)
            d_eta_dE = A * W / (2 * denom ** 2)
            d_eta_dW = A * E / (2 * denom ** 2) if sigma_Width > 0 else 0.0

            var = (
                (d_eta_dA * sigma_A_XY_AHE) ** 2 +
                (d_eta_dS * sigma_S_XY_AHE_art) ** 2 +
                (d_eta_dE * sigma_E_art) ** 2 +
                (d_eta_dW * sigma_Width) ** 2
            )

            sigma_eta = math.sqrt(var)

            return eta, sigma_eta
        def eta2(S_XY_PHE_art, A_XY_PHE, E_art):
            return (S_XY_PHE_art + self.Width * (E_art / 2)) / A_XY_PHE
        def eta3(S_XX_AMR_art, A_XX_AMR, E_art):
            return (S_XX_AMR_art + self.Length * (E_art / 2) ) / A_XX_AMR
        def V_art(E_art, E_art_err):
            S = self.Length * (E_art/2)
            S_err = self.Length * (E_art_err/2)
            return S, S_err

        # Calculating Magnetic properties
        FrequencyScanLong = os.path.join(longitudinal_folder, "FrequencyScan")
        FrequencyScanTrans = os.path.join(transverse_folder, "FrequencyScan")

        if os.path.exists(FrequencyScanLong) is True:
            stfmrLong = STFMR(folder=FrequencyScanLong, voltage_unit='uV', used_fields='Pos', Ms=1e6, d_FM=self.d_FM, d_NM=self.d_NM, Width=self.Width, Length=self.Length)
            stfmrLong.calculate_xiFMR(plot=False)
            MeffLong, MeffLong_err = stfmrLong.Heff_T, stfmrLong.Heff_T_cov
            print(f"Longtidunal xi: {stfmrLong.xi_fmr} ± {stfmrLong.xi_std}. Meff: {stfmrLong.Heff_T} ± {stfmrLong.Heff_T_cov} T")
        else: 
            MeffLong, MeffLong_err = 1.2, 0.01
        
        if os.path.exists(FrequencyScanTrans) is True:
            stfmrTrans = STFMR(folder=FrequencyScanTrans, voltage_unit='uV', used_fields='Pos', Ms=1e6, d_FM=self.d_FM, d_NM=self.d_NM, Width=self.Width, Length=self.Length)            
            stfmrTrans.calculate_xiFMR(plot=False)
            MeffTrans, MeffTrans_err = stfmrTrans.Heff_T, stfmrTrans.Heff_T_cov
            print(f"Transverse xi: {stfmrTrans.xi_fmr} ± {stfmrTrans.xi_std}. Meff: {stfmrTrans.Heff_T} ± {stfmrTrans.Heff_T_cov} T")
        else:
            MeffTrans, MeffTrans_err  = 1.2, 0.01
        Meff, Meff_err = np.average([MeffLong, MeffTrans]), np.average(MeffTrans_err)

        print("1.___Magnetization Dyanamics")
        print(f"Meff average: {Meff} ± {Meff_err}")
        print("Results Resonance Field:")
        print("Transverse measurements, Hres:", np.average(df_transverse['Hres']), "±", np.std(df_transverse['Hres']))
        print("Longitudinal measurements, Hres:",np.average(df_longitudinal['Hres']), "±", np.std(df_longitudinal['Hres']), "\n")
        Hres_avg, Hres_avg_err = np.average(df_transverse['Hres'])*1e-3, np.std(df_transverse['Hres'])*1e-3

        # # Results solving first equation
        E_art_plus1, E_art_plus1_err, E_art_minus1, E_art_minus1_err = solve_E_art_1(A_XY_AHE, S_XY_AHE_art, S_XY_PHE_art, A_XY_PHE, A_XY_AHE_err, S_XY_AHE_art_err, S_XY_PHE_art_err, A_XY_PHE_err)
        xi0, xi0_err = self.FMR_efficiency(S_XX_AMR_art, A_XX_AMR, self.Ms, self.d_FM, self.d_NM, Hres=Hres_avg, Hres_std=Hres_avg_err, Meff=Meff, Meff_std=Meff_err, eta=None, Vs_std=S_XX_AMR_art_err, Va_std=A_XX_AMR_err)
        xi_plus1, xi_plus1_err = self.FMR_efficiency(Vs=0, Va=0, Ms=self.Ms, d_FM=self.d_FM, d_NM=self.d_NM, Hres=Hres_avg, Hres_std=Hres_avg_err, Meff=Meff, Meff_std=Meff_err, eta=eta1(A_XY_AHE, S_XY_AHE_art, E_art_plus1, A_XY_AHE_err, S_XY_AHE_art_err, E_art_plus1_err)[0], eta_std=eta1(A_XY_AHE, S_XY_AHE_art, E_art_plus1, A_XY_AHE_err, S_XY_AHE_art_err, E_art_plus1_err)[1])
        xi_minus1, xi_minus1_err = self.FMR_efficiency(Vs=0, Va=0, Ms=self.Ms, d_FM=self.d_FM, d_NM=self.d_NM, Hres=Hres_avg, Hres_std=Hres_avg_err, Meff=Meff, Meff_std=Meff_err, eta=eta1(A_XY_AHE, S_XY_AHE_art, E_art_minus1, A_XY_AHE_err, S_XY_AHE_art_err, E_art_minus1_err)[0], eta_std=eta1(A_XY_AHE, S_XY_AHE_art, E_art_minus1, A_XY_AHE_err, S_XY_AHE_art_err, E_art_minus1_err)[1])

        print("2a.___Results first equation, AHE-PHE")
        print("Solving the 2nd order equation using the first equation:")
        print(f"E_art_plus: {E_art_plus1} ± {E_art_plus1_err} This gives the following value of eta: ", eta1(A_XY_AHE, S_XY_AHE_art, E_art_plus1, A_XY_AHE_err, S_XY_AHE_art_err, E_art_plus1_err))
        print(f"E_art_plus: {E_art_minus1} ± {E_art_minus1_err} This gives the following value of eta: ", eta1(A_XY_AHE, S_XY_AHE_art, E_art_minus1, A_XY_AHE_err, S_XY_AHE_art_err, E_art_minus1_err), "\n")

        print("2b.___Artifact voltage")
        print(f"S_xx including artifacts: {S_XX_AMR_art} ± {S_XX_AMR_art}")
        print("S_xx_art_plus:", V_art(E_art_plus1, E_art_plus1_err)[0], "So S_XX_AMR_art - artplus =", S_XX_AMR_art - V_art(E_art_plus1, E_art_plus1_err)[0],"ratio with A", V_art(E_art_plus1, E_art_plus1_err)[0]/A_XX_AMR) 
        print("S_xx_art_minus:", V_art(E_art_minus1, E_art_minus1_err)[0], "So S_XX_AMR_art - artminus =",  S_XX_AMR_art - V_art(E_art_minus1, E_art_minus1_err)[0],"ratio with A", V_art(E_art_minus1, E_art_minus1_err)[0]/A_XX_AMR, "\n")

        print("2c.___Calculation of FMR efficiency using: ξ_FMR ≡ η * (e * μ₀ * Mₛ * t_HM * t_FM / ħ) * sqrt(1 + μ₀ * M_eff / B₀)")
        print(f"Using E_art_plus1: {E_art_plus1} ± {E_art_plus1_err} This gives the following value of eta: ", eta1(A_XY_AHE, S_XY_AHE_art, E_art_plus1,A_XY_AHE_err, S_XY_AHE_art_err, E_art_plus1_err))
        print(f"FMR efficiency excluding artifacts by using eta: xi_fmr = {xi_plus1} ± {xi_plus1_err}")
        print(f"FMR efficiency including artifacts: xi_fmr = {xi0} ± {xi0_err}")

        print(f"Using E_art_minus1: {E_art_minus1} ± {E_art_minus1_err} This gives the following value of eta: ", eta1(A_XY_AHE, S_XY_AHE_art, E_art_minus1, A_XY_AHE_err, S_XY_AHE_art_err, E_art_minus1_err))
        print(f"FMR efficiency excluding artifacts by using eta: xi_fmr = {xi_minus1} ± {xi_minus1_err} ")
        print(f"FMR efficiency including artifacts: xi_fmr = {xi0} ± {xi0_err}")

        E_art_plus2, E_art_plus2_err, E_art_minus2, E_art_minus2_err = solve_E_art_2(A_XY_AHE, S_XY_AHE_art, S_XX_AMR_art, A_XX_AMR, A_XY_AHE_err, S_XY_AHE_art_err, S_XX_AMR_art_err, A_XX_AMR_err)
        xi_plus2, xi_plus2_err = self.FMR_efficiency(Vs=0, Va=0, Ms=self.Ms, d_FM=self.d_FM, d_NM=self.d_NM, Hres=Hres_avg, Hres_std=Hres_avg_err, Meff=Meff, Meff_std=Meff_err, eta=eta1(A_XY_AHE, S_XY_AHE_art, E_art_plus2, A_XY_AHE_err, S_XY_AHE_art_err, E_art_plus2_err)[0], eta_std=eta1(A_XY_AHE, S_XY_AHE_art, E_art_plus2, A_XY_AHE_err, S_XY_AHE_art_err, E_art_plus2_err)[1])
        xi_minus2, xi_minus2_err = self.FMR_efficiency(Vs=0, Va=0, Ms=self.Ms, d_FM=self.d_FM, d_NM=self.d_NM, Hres=Hres_avg, Hres_std=Hres_avg_err, Meff=Meff, Meff_std=Meff_err, eta=eta1(A_XY_AHE, S_XY_AHE_art, E_art_minus2, A_XY_AHE_err, S_XY_AHE_art_err, E_art_minus2_err)[0], eta_std=eta1(A_XY_AHE, S_XY_AHE_art, E_art_minus2, A_XY_AHE_err, S_XY_AHE_art_err, E_art_minus2_err)[1])

        # print("3a.___Results first equation, AHE-AMR")
        print(f"E_art_plus: {E_art_plus2} ± {E_art_plus2_err} This gives the following value of eta: ", eta1(A_XY_AHE, S_XY_AHE_art, E_art_plus2, A_XY_AHE_err, S_XY_AHE_art_err, E_art_plus2_err))
        print(f"E_art_plus: {E_art_plus2} ± {E_art_plus2_err} This gives the following value of eta: ", eta1(A_XY_AHE, S_XY_AHE_art, E_art_plus2, A_XY_AHE_err, S_XY_AHE_art_err, E_art_plus2_err))

        # print("3b.___Artifact voltage")
        print(f"S_xx including artifacts: {S_XX_AMR_art} ± {S_XX_AMR_art}")
        print("S_xx_art_plus:", V_art(E_art_plus2, E_art_plus2_err)[0], "So S_XX_AMR_art - artplus =", S_XX_AMR_art - V_art(E_art_plus2, E_art_plus2_err)[0],"ratio with A", V_art(E_art_plus2, E_art_plus2_err)[0]/A_XX_AMR) 
        print("S_xx_art_minus:", V_art(E_art_minus2, E_art_minus2_err)[0], "So S_XX_AMR_art - artminus =",  S_XX_AMR_art - V_art(E_art_minus2, E_art_minus2_err)[0],"ratio with A", V_art(E_art_minus2, E_art_minus2_err)[0]/A_XX_AMR, "\n")

        # print("3c.___Calculation of FMR efficiency using: ξ_FMR ≡ η * (e * μ₀ * Mₛ * t_HM * t_FM / ħ) * sqrt(1 + μ₀ * M_eff / B₀)")
        print(f"Using E_art_plus2: {E_art_plus2} ± {E_art_plus2_err} This gives the following value of eta: ", eta1(A_XY_AHE, S_XY_AHE_art, E_art_plus2,A_XY_AHE_err, S_XY_AHE_art_err, E_art_plus2_err))
        print(f"FMR efficiency excluding artifacts by using eta: xi_fmr = {xi_plus2} ± {xi_plus2_err}")
        print(f"FMR efficiency including artifacts: xi_fmr = {xi0} ± {xi0_err}")

        print(f"Using E_art_minus2: {E_art_minus2} ± {E_art_minus2_err} This gives the following value of eta: ", eta1(A_XY_AHE, S_XY_AHE_art, E_art_minus2, A_XY_AHE_err, S_XY_AHE_art_err, E_art_minus2_err))
        print(f"FMR efficiency excluding artifacts by using eta: xi_fmr = {xi_minus2} ± {xi_minus2_err} ")
        print(f"FMR efficiency including artifacts: xi_fmr = {xi0} ± {xi0_err}")

        # print("4.___Summary: ")
        print(f"Eq. AHE-PHE, +: xi_fmr = {xi_plus1} ± {xi_plus1_err}", "S_xx_art_plus:", V_art(E_art_plus1, E_art_plus1_err)[0], "Ratio:",  V_art(E_art_plus1, E_art_plus1)[0]/A_XY_PHE)
        print(f"Eq. AHE-AMR, -: xi_fmr = {xi_minus1} ± {xi_minus1_err} ,S_xx_art_minus: {V_art(E_art_minus1, E_art_minus1_err)}, ratio with A", V_art(E_art_minus1, E_art_minus1_err)[0]/(A_XY_PHE) , V_art(E_art_minus1, E_art_minus1_err)[0]/(A_XY_PHE) * np.sqrt((V_art(E_art_minus1, E_art_minus1_err)[1]/V_art(E_art_minus1, E_art_minus1_err)[0])**2 + (A_XY_PHE_err/A_XY_PHE)**2))
        print(f"Eq. AHE-AMR, +: xi_fmr = {xi_plus2} ± {xi_plus2_err} ", "S_xx_art_plus:", V_art(E_art_plus2, E_art_plus2_err), "ratio with A", V_art(E_art_plus2, E_art_plus2_err)[0]/(A_XX_AMR))
        print(f"Eq. AHE-AMR, -: xi_fmr = {xi_minus2} ± {xi_minus2_err} ,S_xx_art_minus: {V_art(E_art_minus2, E_art_minus2_err)}, ratio with A", V_art(E_art_minus2, E_art_minus2_err)[0]/(A_XX_AMR) , V_art(E_art_minus2, E_art_minus2_err)[0]/(A_XX_AMR) * np.sqrt((V_art(E_art_minus2, E_art_minus2_err)[1]/V_art(E_art_minus2, E_art_minus2_err)[0])**2 + (A_XX_AMR_err/A_XX_AMR)**2))
        print(f"Angular FMR efficiency including artifacts: xi_fmr ={xi0} ± {xi0_err} ")
        print(f"Conventional 45 scan xi_FMR: {stfmrLong.xi_fmr} ± {stfmrLong.xi_std}")

        print("Calculation of damping torque // 0:")
        print("S_amr_art = (I_rf / 2 alpha omega+) Ramr tau - S_xx_art")
        print("omega+ = omega1 + omega2 = gammaB0 + gamma(B0 + mu0Meff)")


    def _plot_angular_scan(self, df, phi_deg_plot, phi_plot_rad, Vs_func, Va_func, params_Vs, params_Va, label, Vs_components=None, Va_components=None):

        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

        # --- Vs ---
        axes[0].errorbar(df["phi"], df["Vs"], yerr=df["Vsstd"], fmt="o", label="S", zorder=5, markersize=4)
        axes[0].plot(phi_deg_plot, Vs_func(phi_plot_rad, *params_Vs), label="Fit", color="black", linewidth=2)
        if Vs_components:
            for comp in Vs_components:
                axes[0].plot(phi_deg_plot, comp["func"](phi_plot_rad), label=comp["label"], linestyle="--")
        axes[0].set_ylabel("S ($\\mu V$)")
        axes[0].legend(fontsize=10)
        axes[0].grid(True)

        # --- Va ---
        axes[1].errorbar(df["phi"], df["Va"], yerr=df["Vastd"], fmt="s", label="A", zorder=5, color="red", markersize=4)
        axes[1].plot(phi_deg_plot, Va_func(phi_plot_rad, *params_Va), label="Fit", color="black", linewidth=2)
        if Va_components:
            for comp in Va_components:
                axes[1].plot(phi_deg_plot, comp["func"](phi_plot_rad), label=comp["label"], linestyle="--")
        axes[1].set_ylabel("A ($\\mu V$)")
        axes[1].set_xlabel("phi (deg)")
        axes[1].legend(fontsize=10)
        axes[1].grid(True)
        plt.tight_layout()
        # axes[1].legend()

        # --- Vbias ---
        # axes[2].plot(df["phi"], df["C"], "s-", label="Vbias")
        # axes[2].set_ylabel(f"Vbias ({self.voltage_unit})")
        # axes[2].grid(True)
        # axes[2].legend()

        # # --- Hres ---
        # avg_hres = df["Hres"].mean()
        # filtered_df = df[abs(df["Hres"] - avg_hres) <= 15]
        # axes[3].errorbar(
        #     filtered_df["phi"],
        #     filtered_df["Hres"],
        #     yerr=filtered_df["Hresstd"],
        #     fmt="s-",
        #     label="Hres (filtered)"
        # )
        # axes[3].set_ylabel("Hres (mT)")
        # axes[3].set_xlabel("phi (deg)")
        # axes[3].grid(True)
        # axes[3].legend()

        # plt.suptitle(
        #     f"{label} angular scan, f={df['freq'].iloc[0]} GHz",
        #     fontsize=14
        # )
        plt.tight_layout()
        plt.show()


    def plot_Kittel(self, plot=False):
        try:
            if self.used_fields == 'Pos':
                resultspath = os.path.join(self.folder, "fit_results_average_posB.csv")
            elif self.used_fields == 'Neg':
                resultspath = os.path.join(self.folder, "fit_results_average_negB.csv")
        except AttributeError:
            sys.exit("Folder not set. Please initialize STFMR with a folder path to datafiles.")

        if not os.path.exists(resultspath):
            self.average_fitparams()

        df = pd.read_csv(resultspath)
        df_sorted = df.sort_values(by="freq")  # sort by angle
        df_sorted['Hresstd'] = df_sorted['Hresstd'].replace(0, np.max(df_sorted['Hresstd']))

        params, pcov = curve_fit(self.Kittel_formula, abs(df_sorted['Hres']*1e-3), df_sorted['freq'], p0=[1e-3], sigma=df_sorted['Hresstd']*1e-3)

        if self.used_fields == 'Pos':
            H = np.linspace(0, max(df_sorted['Hres']) * 1.1, 200)
        elif self.used_fields == 'Neg':
            H = np.linspace(1.1 * min(df_sorted['Hres']), 0, 200)
        hmin = np.min(H)
        hmax = np.max(H)

        # Plot experimental data
        if plot:
            fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharex=True)
            axes.errorbar(df_sorted['Hres'], df_sorted['freq'], 0, df_sorted['Hresstd'], "o-", label="Data points")

            # Compute fit curve (convert H from mT to T for formula, output from Hz → GHz)
            axes.plot(
                H,
                self.Kittel_formula(abs(H) * 1e-3, *params),  # output in GHz
                '-',
                label=f"Kittel fit: $\\frac{{\\gamma}}{{2 \\pi}} \\sqrt{{|H_{{res}}|(|H_{{res}}| + H_{{eff}})}}$ \n with $\\mu_0 H{{eff}}$ = {params[0]*1e3:.3g} mT"
            )
            axes.set_xlabel("$\\mu_0 H_{res}$ (mT)")
            axes.set_ylabel("Frequency (GHz)")
            axes.grid(True)
            axes.legend(fontsize=14)
            axes.set_ylim(bottom=0, top = 20)
            axes.set_xlim(left=hmin, right=hmax)
            plt.suptitle(f"Fit Results: Resonance Field vs Frequency with {self.used_fields} field sweep", fontsize=14)
            plt.tight_layout()
            plt.show()

            # fig, ax = plt.subplots(3, 1, figsize=(14, 14))
            # ax[0].errorbar(df_sorted['freq'], df_sorted['Va'], yerr=df_sorted['Vastd'], xerr=0, label="Data points Va")
            # ax[0].legend()
            # ax[1].errorbar(df_sorted['freq'], df_sorted['Vs'], yerr=df_sorted['Vsstd'], xerr=0, label="Data points Vs")
            # ax[1].legend()
            # ax[2].plot(df_sorted['freq'], df_sorted['Vs']/df_sorted['Va'], label="Vs/Va")
            # ratio_mean = np.mean(df_sorted['Vs'] / df_sorted['Va'])
            # ratio_std = np.std(df_sorted['Vs'] / df_sorted['Va'])
            # ax[2].axhline(ratio_mean, linestyle='--', label="Mean(Vs/Va)")
            # ax[2].fill_between(
            #     df_sorted['freq'],
            #     ratio_mean - ratio_std,
            #     ratio_mean + ratio_std,
            #     alpha=0.2,
            #     label="Std(Va/Vs)"
            # )
            # ax[2].legend()

        self.Heff_T, self.Heff_T_cov = params[0], np.sqrt(pcov[0])
        return df_sorted
    
    def plot_Kittel2(self, plot=False):
        try:
            if self.used_fields == 'Pos':
                resultspath = os.path.join(self.folder, "fit_results_average_posB.csv")
            elif self.used_fields == 'Neg':
                resultspath = os.path.join(self.folder, "fit_results_average_negB.csv")
        except AttributeError:
            sys.exit("Folder not set. Please initialize STFMR with a folder path to datafiles.")

        if not os.path.exists(resultspath):
            self.average_fitparams()

        df = pd.read_csv(resultspath)
        df_sorted = df.sort_values(by="freq")  # sort by angle
        df_sorted['Hresstd'] = df_sorted['Hresstd'].replace(0, np.max(df_sorted['Hresstd']))

        params, pcov = curve_fit(self.Kittel_formula2, abs(df_sorted['Hres']*1e-3), df_sorted['freq'], bounds=([0, 0], [np.inf, np.inf]), p0=[1, 1], sigma=df_sorted['Hresstd']*1e-3)

        if self.used_fields == 'Pos':
            H = np.linspace(0, max(df_sorted['Hres']) * 1.1, 200)
        elif self.used_fields == 'Neg':
            H = np.linspace(1.1 * min(df_sorted['Hres']), 0, 200)
        hmin = np.min(H)
        hmax = np.max(H)
        self.Heff_T, self.Heff_T_cov = params[0], np.sqrt(pcov[0])
        self.Hk, self.Hk_cov = params[1], np.sqrt(pcov[1])
        # Plot experimental data
        if plot:
            fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharex=True)
            axes.errorbar(df_sorted['Hres'], df_sorted['freq'], 0, df_sorted['Hresstd'], "o-", label="Data points")

            # Compute fit curve (convert H from mT to T for formula, output from Hz → GHz)
            axes.plot(
                H,
                self.Kittel_formula2(abs(H) * 1e-3, *params),  # output in GHz
                '-',
                label=f"Kittel fit: $\\frac{{\\gamma}}{{2 \\pi}} \\sqrt{{(|H_{{res}}| + Hk)(|H_{{res}}| + Hk + H_{{eff}})}}$ \n with $\\mu_0 H{{eff}}$ = {params[0]*1e3:.3g} mT and Hk = {params[1]:.3g}"
            )
            axes.set_xlabel("$\\mu_0 H_{res}$ (mT)")
            axes.set_ylabel("Frequency (GHz)")
            axes.grid(True)
            axes.legend(fontsize=14)
            axes.set_ylim(bottom=0, top = 20)
            axes.set_xlim(left=hmin, right=hmax)
            plt.suptitle(f"Fit Results: Resonance Field vs Frequency with {self.used_fields} field sweep", fontsize=14)
            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots(3, 1, figsize=(14, 14))
            ax[0].errorbar(df_sorted['freq'], df_sorted['Va'], yerr=df_sorted['Vastd'], xerr=0, label="Data points Va")
            ax[0].legend()
            ax[1].errorbar(df_sorted['freq'], df_sorted['Vs'], yerr=df_sorted['Vsstd'], xerr=0, label="Data points Vs")
            ax[1].legend()
            ax[2].plot(df_sorted['freq'], df_sorted['Vs']/df_sorted['Va'], label="Vs/Va")
            ratio_mean = np.mean(df_sorted['Vs'] / df_sorted['Va'])
            ratio_std = np.std(df_sorted['Vs'] / df_sorted['Va'])
            ax[2].axhline(ratio_mean, linestyle='--', label="Mean(Vs/Va)")
            ax[2].fill_between(
                df_sorted['freq'],
                ratio_mean - ratio_std,
                ratio_mean + ratio_std,
                alpha=0.2,
                label="Std(Va/Vs)"
            )
            ax[2].legend()
            print('Hk=', params[0], 'Meff=', params[1])
        return df_sorted
    
    def fit_kittel(self, mu0H_FMR, frequencies, mu0H_FMR_std=0): #-+
        initial_guesses = [1e-5]  # Initial guesses for gamma, mu0M_eff
        popt, pcov = curve_fit(self.kittel_function, mu0H_FMR, frequencies, p0=initial_guesses, sigma=mu0H_FMR_std)
        
        # Extract uncertainties
        perr = np.sqrt(np.diag(pcov))
        
        # Save the parameters
        kittel_params = {
            'mu0M_eff': ufloat(popt[0], perr[0]),
        }
        
        return popt, pcov, kittel_params

    # Step 1: Define the Kittel function
    def kittel_function(self, mu0H_FMR, mu0M_eff):
        #!!!! a = gamma/2pi (GHz/mT); It is material dependent, change accordingly after checking literature!!!!
        a = 0.0294  #GHz/mT
        return  a * np.sqrt(mu0H_FMR * (mu0H_FMR + mu0M_eff))  
    

    def plot_linewidth(self, plot=True):
        try:
            if self.used_fields == 'Pos':
                resultspath = os.path.join(self.folder, "fit_results_average_posB.csv")
            elif self.used_fields == 'Neg':
                resultspath = os.path.join(self.folder, "fit_results_average_negB.csv")
        except AttributeError:
            sys.exit("Folder not set. Please initialize STFMR with a folder path to datafiles.")
        

        df = pd.read_csv(resultspath)
        df_sorted = df.sort_values(by="freq")  # sort by angle
        errs = df_sorted['Deltastd'].fillna(df_sorted['Deltastd'].max())  # remove NaNs, large uncertainty → low weight
        df_sorted['Deltastd'] = df_sorted['Deltastd'].replace(0, np.max(df_sorted['Deltastd']))

        params, pcov = curve_fit(self.linewidth_formula, df_sorted["freq"]*1e9, df_sorted['Delta']*1e-3, p0=[1/self.mu_0, 0.1], sigma=df_sorted['Deltastd']*1e-3)
        errs = np.sqrt(np.diag(pcov))
        # print(params)
        freq = np.linspace(0, max(df_sorted["freq"])*1.2, 200)

        # --- Plot H vs freq ---
        if plot == True:
            fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
            axes.errorbar(df_sorted["freq"], df_sorted["Delta"], df_sorted["Deltastd"], 0, "o-", label="Sample data")
            axes.plot(freq, self.linewidth_formula(freq*1e9, *params)*1e3, '-', label=f"Linewidth fit: $\\mu_0 \\Delta H_0 + \\frac{{2 \\pi \\alpha f}}{{\\gamma}}$ \n with inhomogeneous broadening $\\mu_0 \\Delta H_0$ = {params[0]*self.mu_0*1e3:.3g} mT \n and damping factor $\\alpha = $ {params[1]:.3g}") #
            axes.set_xlabel("Frequency (GHz)")
            axes.set_ylabel(f"$\\mu \\Delta H$ (mT)")
            axes.grid(True)
            axes.legend(fontsize=14)
            axes.set_xlim(right=18)
            # axes.set_ylim(bottom=0, top=8)
            plt.suptitle("Fit Results: $\\mu \\Delta H_0$ vs frequency", fontsize=14)
            plt.tight_layout()
            plt.show()

        self.alpha, self.DeltaH0 = params[1], params[0]
        self.alpha_cov, self.DeltaH0_cov = errs[1], errs[0]
        return df_sorted

    def plot_power(self, method='Average_fitparams'):
        sample_folder = self.folder
        resultspath_pos = os.path.join(sample_folder, "fit_results_average_posB.csv")
        resultspath_pos_avg = os.path.join(sample_folder, "Averaged_Traces_posB", "fit_results_average_posB.csv")

        self.used_fields = 'Pos'
        if not os.path.exists(resultspath_pos):
            self.average_fitparams(plot=False, folder=sample_folder)
        if not os.path.exists(resultspath_pos_avg):
            self.analyze_traces()
            self.average_fitparams(plot=False, folder=Path(resultspath_pos_avg).parent)

        if method == 'Average_fitparams':
            df = pd.read_csv(resultspath_pos)
        if method == 'Average_measurements':
            df = pd.read_csv(resultspath_pos_avg)
        # print(df.head())
        df_filtered = df   #[df['pow_'] <= 3]

        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        df_sorted = df_filtered.sort_values(by="pow_")  # sort by power

        power = np.array(df_sorted["pow_"]) + 27
        
        milliwatts = 10 ** (power / 10)  # Convert dBm to mW
        mask = milliwatts >= -5

        xmin = milliwatts.min()
        xmax = milliwatts.max()

        for ax in axes:
            ax.set_xlim(-10, xmax*1.1)
            
        # --- Plot H vs freq ---
        axes[0].errorbar(milliwatts[mask], df_sorted["Vs"][mask], df_sorted["Vsstd"][mask], 0, "o-", label="S")
        axes[0].errorbar(milliwatts[mask], df_sorted["Va"][mask], df_sorted["Vastd"][mask], 0, "o-", label="A")
        # axes[0].errorbar(milliwatts, df_sorted["C"], df_sorted["Cstd"], 0, "o-", label="C")
        axes[0].set_xlabel("Power (mW)", fontsize=14)
        axes[0].set_ylabel(f"Voltage ($\\mu$V)", fontsize=14)
        axes[0].grid(True)
        axes[0].legend(fontsize=14)
        # axes[0].set_xscale("log")
        axes[0].axvline(10 ** (27/10), color='gray', linestyle='--')
        self.plot_Kittel(plot=False)

        xi_fmr, xi_fmr_std = self.FMR_efficiency(Vs=df_sorted["Vs"], Va=df_sorted["Va"], Ms=self.Ms, d_FM=self.d_FM, d_NM=self.d_NM, Hres=df_sorted["Hres"]*1e-3, Meff=self.Heff_T, eta=None, Vs_std=df_sorted["Vsstd"], Va_std=df_sorted["Vastd"], Hres_std=df_sorted["Hresstd"]*1e-3, Meff_std=self.Heff_T_cov)

        # ---- 3. xi_FMR ----
        axes[1].errorbar(milliwatts, df_sorted["Hres"], df_sorted["Hresstd"], 0, "o-", label="Hres")
        axes[1].set_xlabel("Power (mW)", fontsize=14)
        axes[1].grid(True)
        axes[1].set_ylabel("$H_{res}$ (mT)", fontsize=14)

        # axes[1].legend(fontsize=14)
        # axes[2].set_xscale("log")
        axes[1].axvline(10 ** (27/10), color='gray', linestyle='--')
        plt.suptitle(f"{self.sample_name}", fontsize=14)
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1,1)
        axes.errorbar(
            milliwatts[mask],
            xi_fmr[mask],
            yerr=xi_fmr_std[mask],
            fmt='o-',
            label='xi_FMR'
        )
        axes.set_xlabel("Power (mW)", fontsize=14)
        axes.set_ylabel(f"$\\xi_{{FMR}}$ ", fontsize=14)
        axes.set_xscale("log")
        axes.grid(True)
        axes.set_xlim(left=0)
        plt.suptitle(f"{self.sample_name}", fontsize=14)

        # axes.legend(fontsize=14)
        # axes[1].plot(power, (df_sorted["Vs"]/df_sorted["Va"])* (1/np.sqrt(df_sorted["Hres"])), "o-", label="Vs/Va")
        # axes.legend()
        axes.axvline(10 ** (27/10), color='gray', linestyle='--')
        # self.plot_general(df_sorted, 'pow_', 'C')
        return


    def DCScan(self, method='Average_fitparams', Meff=1.12, Meff_err=None):
        DC_folder = self.folder

        resultspath_pos = os.path.join(DC_folder, "fit_results_average_posB.csv")
        resultspath_neg = os.path.join(DC_folder, "fit_results_average_negB.csv")

        resultspath_pos_avg = os.path.join(DC_folder, "Averaged_Traces_posB", "fit_results_average_posB.csv")
        resultspath_neg_avg = os.path.join(DC_folder, "Averaged_Traces_negB", "fit_results_average_negB.csv")

        self.used_fields = 'Pos'
        if not os.path.exists(resultspath_pos):
            self.average_fitparams(plot=False, folder=DC_folder)
        if not os.path.exists(resultspath_pos_avg):
            self.analyze_traces()
            self.average_fitparams(plot=False, folder=Path(resultspath_pos_avg).parent)

        self.used_fields = 'Neg'
        if not os.path.exists(resultspath_neg):
            self.average_fitparams(plot=False, folder=DC_folder)
        if not os.path.exists(resultspath_neg_avg):
            self.analyze_traces()
            self.average_fitparams(plot=False, folder=Path(resultspath_neg_avg).parent)

        if method == 'Average_fitparams':
            df_pos = pd.read_csv(resultspath_pos)
            df_neg = pd.read_csv(resultspath_neg)
        if method == 'Average_measurements':
            df_pos = pd.read_csv(resultspath_pos_avg)
            df_neg = pd.read_csv(resultspath_neg_avg)

        fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
        sample = Path(DC_folder).parent.name   # one level higher
        df_pos = df_pos.sort_values(by="dc")  # sort by dc current
        df_neg = df_neg.sort_values(by="dc")  # sort by dc current        

        df_pos["J_Pt"] = self.calculate_current_density_from_DC(df_pos["dc"]*1e-3)
        df_neg["J_Pt"] = self.calculate_current_density_from_DC(df_neg["dc"]*1e-3)

        angle, frequency, Hext, Hext_err = np.average(df_pos['phi']), np.average(df_pos["freq"])*1e9, np.average(df_pos["Hres"]), np.std(df_pos["Hres"])

        # Define a common errorbar style for consistency
        err_style = dict(
            fmt="+-",             # circle markers, no connecting line
            capsize=4,           # small caps on error bars
            elinewidth=1,        # thin error bar lines
            linewidth=0,
            markersize=7,        # marker size
        )

        # if sample == 'DK158':
        #     cutoff = 0
        # else:
        #     cutoff = 0
        cutoff = 0
        params_neg, pcov_neg = curve_fit(
            self.linewidth_DC_formula,
            df_neg['J_Pt'],
            df_neg['Delta'] * 1e-3,
            p0=[5, 1],
            sigma=df_neg['Deltastd'] * 1e-3   # scale sigma too if y was scaled
        )

        params_pos, pcov_pos = curve_fit(
            self.linewidth_DC_formula,
            df_pos['J_Pt'],
            df_pos['Delta'] * 1e-3,
            p0=[5, 1],
            sigma=df_pos['Deltastd'] * 1e-3
        )

        b_neg, m_neg = params_neg
        b_pos, m_pos = params_pos

        sigma_b_neg, sigma_m_neg = np.sqrt(np.diag(pcov_neg))
        sigma_b_pos, sigma_m_pos = np.sqrt(np.diag(pcov_pos))

        # intersection in ORIGINAL x units: A/m^2
        den = (m_neg - m_pos)
        x_cross = (b_pos - b_neg) / den

        dx_dbp = 1 / den
        dx_dbn = -1 / den
        dx_dmp = (b_pos - b_neg) / den**2
        dx_dmn = -(b_pos - b_neg) / den**2

        sigma_x = np.sqrt(
            (dx_dbp * sigma_b_pos)**2 +
            (dx_dbn * sigma_b_neg)**2 +
            (dx_dmp * sigma_m_pos)**2 +
            (dx_dmn * sigma_m_neg)**2
        )

        # y_cross in ORIGINAL y units used in fit
        y_cross = m_neg * x_cross + b_neg

        sigma_y = np.sqrt(
            (x_cross * sigma_m_neg)**2 +
            sigma_b_neg**2 +
            (m_neg * sigma_x)**2
        )

        # only rescale for display
        x_cross_plot = x_cross * 1e-10
        sigma_x_plot = sigma_x * 1e-10

        y_cross_alpha = y_cross * (self.gamma / (2 * np.pi * frequency))
        sigma_y_alpha = sigma_y * (self.gamma / (2 * np.pi * frequency))

        print("Intersection J_Pt x:", x_cross_plot)
        print("Intersection J_Pt error:", sigma_x_plot)
        print("Intersection alpha y:", y_cross_alpha)
        print("Intersection alpha error:", sigma_y_alpha)

        J_Pt = self.create_fit_array(min(df_pos['J_Pt']), max(df_pos['J_Pt']), num_points=200)

        # --- Plot Delta vs dc ---
        # axes[0].errorbar(
        #     df_neg["J_Pt"].iloc[cutoff:],
        #     df_neg["Delta"].iloc[cutoff:],
        #     df_neg["Deltastd"].iloc[cutoff:],
        #     label="Delta negative field",
        #     color="tab:red",
        #     **err_style
        # )
        # axes[0].errorbar(
        #     df_pos["J_Pt"].iloc[cutoff:],
        #     df_pos["Delta"].iloc[cutoff:],
        #     df_pos["Deltastd"].iloc[cutoff:],
        #     label="Delta positive field",
        #     color="tab:blue",
        #     **err_style
        # )
        # axes[0].plot(J_Pt, self.linewidth_DC_formula(J_Pt, *params_neg)*1e3, label=f'Fit for negative field, slope = {params_neg[1]:.2g}', color= 'r', linestyle='--') # : $\mu_0 \Delta H_0 + I_{DC} \kappa$
        # axes[0].plot(J_Pt, self.linewidth_DC_formula(J_Pt, *params_pos)*1e3, label=f'Fit for positive field, slope = {params_pos[1]:.2g}' , color= 'b', linestyle='--') #: $\mu_0 \Delta H_0 + I_{DC} \kappa$

        # axes[0].set_xlabel("J_Pt (A/m2)")
        # # axes[0].set_ylim(4, 8)
        # axes[0].set_ylabel("Delta (mT)")
        # axes[0].set_title("Delta vs dc")
        # axes[0].grid(True, linestyle="--", alpha=0.6)
        # axes[0].legend()

        axes.errorbar(
            df_neg["J_Pt"].iloc[cutoff:]*1e-10,
            df_neg["Delta"].iloc[cutoff:]*1e-3 * (self.gamma / (2 * np.pi * frequency)),
            df_neg["Deltastd"].iloc[cutoff:]*1e-3 * (self.gamma / (2 * np.pi * frequency)),
            label="Data negative field",
            color="tab:red",
            **err_style
        )
        axes.errorbar(
            df_pos["J_Pt"].iloc[:]*1e-10,
            df_pos["Delta"].iloc[:] * 1e-3 * (self.gamma / (2 * np.pi * frequency)),
            df_pos["Deltastd"].iloc[:] * 1e-3 * (self.gamma / (2 * np.pi * frequency)),
            label="Data positive field",
            color="tab:blue",
            **err_style
        )

        alpha_factor = self.gamma / (2 * np.pi * frequency)

        slope_neg_alpha = params_neg[1] * alpha_factor
        slope_pos_alpha = params_pos[1] * alpha_factor

        axes.plot(
            J_Pt*1e-10,
            self.linewidth_DC_formula(J_Pt, *params_neg) * alpha_factor,
            label=f'Fit for negative field, $\\Delta \\alpha / \\Delta J_{{Pt}}$ = {slope_neg_alpha:.2g}',
            color='r',
            linestyle='--'
        )

        axes.plot(
            J_Pt*1e-10,
            self.linewidth_DC_formula(J_Pt, *params_pos) * alpha_factor,
            label=f'Fit for positive field, $\\Delta \\alpha / \\Delta J_{{Pt}}$ = {slope_pos_alpha:.2g}',
            color='b',
            linestyle='--'
        )
        axes.set_xlabel(r"$J_{Pt} \times 10^{10}\ (\mathrm{A/m^2})$",  fontsize=12)        
        axes.set_ylabel(f"$\\alpha$", fontsize=12)
        # axes.set_title("Damping factor versus longitudinal direct current density", fontsize=14)
        axes.grid(True, linestyle="--", alpha=0.6)
        axes.legend()

        # --- Plot Vs vs dc ---
        # axes[2].errorbar(
        #     df_pos["J_Pt"].iloc[1:],
        #     df_pos["Vs"].iloc[1:],
        #     df_pos["Vsstd"].iloc[1:],
        #     label="Vs positive field",
        #     color="tab:blue",
        #     **err_style
        # )
        # axes[2].errorbar(
        #     df_neg["J_Pt"].iloc[1:],
        #     df_neg["Vs"].iloc[1:],
        #     df_neg["Vsstd"].iloc[1:],
        #     label="Vs negative field",
        #     color="tab:red",
        #     **err_style
        # )
        # # --- Plot Va vs dc ---
        # axes[2].errorbar(
        #     df_pos["J_Pt"].iloc[1:],
        #     df_pos["Va"].iloc[1:],
        #     df_pos["Vastd"].iloc[1:],
        #     label="Va positive field",
        #     color="tab:green",
        #     **err_style
        # )
        # axes[2].errorbar(
        #     df_neg["J_Pt"].iloc[1:],
        #     df_neg["Va"].iloc[1:],
        #     df_neg["Vastd"].iloc[1:],
        #     label="Va negative field",
        #     color="tab:green",
        #     **err_style
        # )
        # axes[2].set_xlabel("J_Pt (A/m2)")
        # axes[2].set_ylabel(f"Vs ({self.voltage_unit})")
        # axes[2].set_title("Vs vs dc")
        # axes[2].grid(True, linestyle="--", alpha=0.6)
        # axes[2].legend()

        # fig.suptitle(f"{sample}", fontsize=14, fontweight="bold")
        # plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

        print("----- Analysis of FMR efficiency 2----- Paper DOI: file:///C:/Users/chris/Documents/DocumentsAndLiterature/Master%20Research%20Project/Literature/ST%20FMR/STFMR_for%20Report.pdf")
        print("Delta_H = (2*pi*f/gamma) * alpha + (2*pi*f/gamma) * (sin(theta) / ((H_ext + 2*pi*M_eff) * mu_0 * M_s * t)) * (hbar / 2*e) * J_c * theta_sh")
        print(f"Using variables: mu0 = {self.mu_0} T·m/A, Ms = {self.Ms} A/m, t_FM = {self.d_FM} m, theta = {angle} degrees, f = {frequency} Hz, mu0 Meff = 1.12 T, Hext = {Hext*1e-3} T, slope = {params_pos[1]} mT/mA (T/A)")
        print("Slope = (2*pi*f/gamma) * (sin(theta) / ((H_ext + 2*pi*M_eff) * mu_0 * M_s * t)) * (hbar / 2*e) * theta_sh")
        print("theta_sh = ((H_ext + 2*pi*M_eff) * mu_0 * M_s * t) / sin(theta) * (gamma / 2*pi*f) * Slope * (2 e /  hbar)")
        
        theta_sh_pos = (
            params_pos[1] / self.mu_0
            * (self.gamma / (2 * np.pi * frequency))
            * (2 * self.e / self.hbar)
            * (self.mu_0 * self.Ms * self.d_FM / np.sin(np.deg2rad(angle)))
            * (Hext*1e-3 + 0.5 * Meff) 
        )
        p_pos = params_pos[1]
        sigma_p_pos = np.sqrt(pcov_pos[1, 1])
        A = (
            (self.gamma / (2 * np.pi * frequency))
            * (2 * self.e / self.hbar)
            * (self.Ms * self.d_FM / np.sin(np.deg2rad(angle)))
        )
        B = Hext * 1e-3 + 0.5 * Meff

        theta_sh_pos_err = np.sqrt(
            (A * B * sigma_p_pos)**2 +
            (A * p_pos * 1e-3 * Hext_err)**2 +
            (A * p_pos * 0.5 * Meff_err)**2
        )

        # print("Term slope:", params_pos[1], "gamma/2pif:", (self.gamma / (2 * np.pi * frequency)), "2e/hbar:", (2 * self.e / self.hbar), "mu0Ms*t/sin(theta):", (self.mu_0 * self.Ms * self.d_FM / np.sin(np.deg2rad(angle))), "(Hext + 2pi*Meff):", (Hext*1e-3 + 2 * np.pi * Meff))
        
        print("found theta_sh for positive field: ", theta_sh_pos, "±", theta_sh_pos_err)
        theta_sh_neg = (
            params_neg[1] / self.mu_0
            * (self.gamma / (2 * np.pi * frequency))
            * (2 * self.e / self.hbar)
            * (self.mu_0 * self.Ms * self.d_FM / np.sin(np.deg2rad(angle)+np.pi))
            * (Hext*1e-3 + 0.5 * Meff) 
        )
        p_neg = params_neg[1]
        sigma_p_neg = np.sqrt(pcov_neg[1, 1])
        A = (
            (self.gamma / (2 * np.pi * frequency))
            * (2 * self.e / self.hbar)
            * (self.Ms * self.d_FM / np.sin(np.deg2rad(angle)))
        )
        B = Hext * 1e-3 + 0.5 * Meff

        theta_sh_neg_err = np.sqrt(
            (A * B * sigma_p_neg)**2 +
            (A * p_neg * 1e-3 * Hext_err)**2 +
            (A * p_neg * 0.5 * Meff_err)**2
        )

        # print("Term slope:", params_neg[1], "gamma/2pif:", (self.gamma / (2 * np.pi * frequency)), "2e/hbar:", (2 * self.e / self.hbar), "mu0Ms*t/sin(theta):", (self.mu_0 * self.Ms * self.d_FM / np.sin(np.deg2rad(angle))), "(Hext + 2pi*Meff):", (Hext*1e-3 + 2 * np.pi * Meff))
        print("found theta_sh for negative field: ", theta_sh_neg, "±", theta_sh_neg_err)
        print("Intersection J_Pt x:", x_cross_plot)
        print("Intersection J_Pt error:", sigma_x_plot)
        print("Intersection alpha y:", y_cross_alpha)
        print("Intersection alpha error:", sigma_y_alpha)

    def calculate_xiFMR(self, plot=False):
        # Check for required data and extract it
        try:
            if self.used_fields == 'Pos':
                resultspath = os.path.join(self.folder, "fit_results_average_posB.csv")
            elif self.used_fields == 'Neg':
                resultspath = os.path.join(self.folder, "fit_results_average_negB.csv")
        except AttributeError:
            sys.exit("Folder not set. Please initialize STFMR with a folder path to datafiles.")
        
        if not os.path.exists(resultspath):
            self.average_fitparams()

        df = pd.read_csv(resultspath)
        df_sorted = df.sort_values(by="freq").reset_index(drop=True)  # sort by frequency

        if not hasattr(self, "Heff_T") or self.Heff_T is None:
            self.plot_Kittel(plot=False) 

        if not hasattr(self, "Alpha") or self.alpha is None:
            self.plot_linewidth(plot=False)

        allowed_freqs = [9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16] 

        df_to_save = df_sorted[
            (df_sorted['freq'].isin(allowed_freqs)) & 
            (df_sorted['phi'].round() == 45)
        ].copy()

        self.freq_ = df_to_save['freq'].to_numpy()
        self.Vs_ = df_to_save['Vs'].to_numpy()
        self.Va_ = df_to_save['Va'].to_numpy()
        self.Hres_ = df_to_save['Hres'].to_numpy() * 1e-3  # in T

        xi_list = []
        xi_std_list = []
        # Error propagation made by ChatGPT
        for _, row in df_to_save.iterrows():
            xi, xi_std = self.FMR_efficiency(
                Vs=row['Vs'],
                Va=row['Va'],
                Ms=self.Ms,
                d_FM=abs(self.d_FM),
                d_NM=abs(self.d_NM),
                Hres=row['Hres'] * 1e-3,     # T
                Meff=self.Heff_T,
                Vs_std=row.get('Vsstd', 0.0),
                Va_std=row.get('Vastd', 0.0),
                Hres_std=row.get('Hresstd', 0.0) * 1e-3,
                Meff_std=self.Heff_T_cov
            )

            xi_list.append(float(xi))
            xi_std_list.append(float(xi_std))

        df_to_save['xi_FMR'] = xi_list
        df_to_save['xi_FMR_std'] = xi_std_list
        
        self.xi_fmr = np.mean(xi_list)
        self.xi_std = self.uncorrelated_errors(np.mean(xi_std_list), np.std(xi_list))
        self.Vs_mean, self.Va_mean = np.mean(df_to_save['Vs']), np.mean(df_to_save['Va'])
        # print("Found xi_FMR =", self.xi_fmr, "±", self.xi_std)

        # Ensure no leftover xi columns from previous merges
        df = df.drop(columns=[c for c in df.columns if c.startswith('xi_FMR')], errors='ignore')

        # Merge cleanly
        df = df.merge(df_to_save[['freq', 'xi_FMR', 'xi_FMR_std']], on='freq', how='left')
        # print(df)
        df.to_csv(resultspath, index=False)

        if plot==True: 
            fig, axs = plt.subplots(3, 1, figsize=(12, 14))
            axs = axs.flatten()   # easier indexing

            # ---- 1. Va and Vs ----
            axs[0].plot(df_to_save['freq'], df_to_save['Va'], 'o-', label='Va')
            axs[0].plot(df_to_save['freq'], df_to_save['Vs'], 'o-', label='Vs')
            axs[0].set_xlabel('Frequency (GHz)')
            axs[0].set_ylabel('Amplitude (arb.)')
            axs[0].set_title('Va and Vs vs Frequency')
            axs[0].grid(True)
            axs[0].legend()

            # ---- 2. Vs / Va ----
            ratio = df_to_save['Vs'] / df_to_save['Va']
            axs[1].plot(df_to_save['freq'], ratio, 'o-')
            axs[1].set_xlabel('Frequency (GHz)')
            axs[1].set_ylabel('Vs / Va')
            axs[1].set_title('Vs/Va vs Frequency')
            axs[1].grid(True)

            # ---- 3. xi_FMR ----
            axs[2].errorbar(
                df_to_save['freq'],
                df_to_save['xi_FMR'],
                yerr=df_to_save['xi_FMR_std'],
                fmt='o-',
                label='xi_FMR'
            )
            
            axs[2].set_xlabel('Frequency (GHz)')
            axs[2].set_ylabel('FMR efficiency ($\\xi_{FMR}$)')
            axs[2].set_title('$\\xi_{FMR}$ vs Frequency')
            axs[2].grid(True)

            fig.suptitle(f"FMR Analysis Overview — Sample: {self.folder}", fontsize=16)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    def read_and_print_csv(self, name):
        path = os.path.join(self.folder, name)
        df = pd.read_csv(path, delimiter=',', header=0)
        print(df)

    # ---------------- FIT MODELS AND EQUATIONS ----------------
    def create_fit_array(self, xmin, xmax, num_points=100):
        if xmin < 0:
            x_fit_min = xmin * 1.1
        if xmax < 0:
            x_fit_max = xmax * 0.9
        if xmin >= 0: 
            x_fit_min = xmin * 0.9
        if xmax >= 0:
            x_fit_max = xmax * 1.1
        return np.linspace(x_fit_min, x_fit_max, num_points)

    def uncorrelated_errors(self, *errors):
        errors = np.array(errors, dtype=float)
        # Convert NaNs to zero
        errors = np.nan_to_num(errors, nan=0.0)
        return np.sqrt(np.sum(errors**2))
    
    def dbm_to_watts(self, dbm):
        return 10 ** ((dbm - 30) / 10)
    
    def calculate_current_density(self, power_dbm, R=50, amplifier_db=27, rho_NM=10.6e-8, rho_FM=150e-8):
        # Calculate current amplitude from power in dBm
        power_watts = self.dbm_to_watts(power_dbm+amplifier_db)
        current_amplitude = np.sqrt(2 * power_watts / R)  # Peak current in Amperes

        # Calculate current density (A/m^2), using geometry and material parameters
        width = self.Width * 1e-6 
        thickness_FM, thickness_NM = self.d_FM * 1e-9, self.d_NM * 1e-9  

        f_NM = (thickness_NM/rho_NM) / ((thickness_NM/rho_NM) + (thickness_FM/rho_FM))  
        I_NM = current_amplitude * f_NM

        current_density = I_NM / (width * thickness_NM)  
        return current_density, I_NM, current_amplitude
    
    def calculate_current_density_from_DC(self, I, rho_NM=30.6e-8, rho_FM=125e-8):
        width = self.Width 
        thickness_FM, thickness_NM = self.d_FM , self.d_NM 
        
        f_NM = (thickness_NM/rho_NM) / ((thickness_NM/rho_NM) + (thickness_FM/rho_FM)) 

        J_Pt = (I * f_NM) / (width * thickness_NM)
        return J_Pt

    def Kittel_formula(self, Hres, Heff):
        return (28.4) * np.sqrt((Hres) * (Hres + Heff))
    
    def Kittel_formula2(self, Hres, Heff, Hk):
        return (28.4) * np.sqrt((Hres + Hk) * (Hres + Hk + Heff))

    def linewidth_formula(self, f, DeltaH0, alpha):
        return self.mu_0*DeltaH0 + (np.pi * 2 * f * alpha)/self.gamma
    
    def linewidth_DC_formula(self, J_Pt, DeltaH0, kappa):
        return DeltaH0 + J_Pt * kappa
    
    def tau_DL(self, xi_DL, Je, Ms, t_FM):
        """Damping-like torque"""
        return xi_DL * self.mu_B * Je / (self.e * Ms * t_FM)

    def tau_FL(self, xi_FL, Je, Ms, t_FM):
        """Field-like torque"""
        return xi_FL * self.mu_B * Je / (self.e * Ms * t_FM)

    def tau_Oe(self, mu_0, gamma, Je, t_HM):
        """Oersted torque"""
        return mu_0 * gamma * Je * t_HM / 2
    
    # def FMR_efficiency(self, Vs, Va, Ms, d_FM, d_NM, Hres, Meff=1, eta=None):
    #     if eta is None:
    #         eta = Vs/Va
    #     # print('Vs/Va', eta,'Vs', Vs, 'Va', Va, 'Hres', (Hres), 'Meff', Meff, 'Ms', Ms, 'd_FM', d_FM, 'd_NM', d_NM)
    #     return eta * (self.e/self.hbar) * (self.mu_0 * Ms * d_FM * d_NM ) * np.sqrt(1+(((Meff))/Hres))

    # def FMR_efficiency(
    #     self,
    #     Vs, Va, Ms, d_FM, d_NM, Hres, Hres_std=None, Meff=1, Meff_std=None,
    #     eta=None, eta_std=None,
    #     Vs_std=None, Va_std=None
    # ):

    #     if Ms > 1e3:
    #         Ms = self.mu_0 * Ms

        
    #     eta_from_ratio = (eta is None)
    #     if eta_from_ratio:
    #         eta = Vs / Va

    #     C = (self.e / self.hbar) * (Ms * d_FM * d_NM)
    #     sqrt_term = np.sqrt(1 + Meff / Hres)
    #     xi = eta * C * sqrt_term

    #     # Collect variance contributions only from provided uncertainties
    #     xi_var = 0.0
    #     has_error = False

    #     # Common derivatives
    #     dxi_dHres = (
    #         C * eta *
    #         0.5 * (1 + Meff / Hres) ** (-0.5) *
    #         (-Meff / Hres**2)
    #     )

    #     dxi_dMeff = (
    #         C * eta *
    #         0.5 * (1 + Meff / Hres) ** (-0.5) *
    #         (1 / Hres)
    #     )

    #     # eta handling
    #     if eta_from_ratio:
    #         if Vs_std is not None:
    #             dxi_dVs = xi / Vs
    #             xi_var += (dxi_dVs * Vs_std) ** 2
    #             has_error = True

    #         if Va_std is not None:
    #             dxi_dVa = -xi / Va
    #             xi_var += (dxi_dVa * Va_std) ** 2
    #             has_error = True
    #     else:
    #         if eta_std is not None:
    #             dxi_deta = xi / eta
    #             xi_var += (dxi_deta * eta_std) ** 2
    #             has_error = True

    #     # Common optional contributions
    #     if Hres_std is not None:
    #         xi_var += (dxi_dHres * Hres_std) ** 2
    #         has_error = True

    #     if Meff_std is not None:
    #         xi_var += (dxi_dMeff * Meff_std) ** 2
    #         has_error = True

    #     if has_error:
    #         xi_std = np.sqrt(xi_var)
    #         return xi, xi_std
    #     else:
    #         return xi


    def FMR_efficiency(
        self,
        Vs, Va, Ms, d_FM, d_NM, Hres, Hres_std=None, Meff=1, Meff_std=None,
        eta=None, eta_std=None,
        Vs_std=None, Va_std=None,
        debug=False
    ):
        """
        Returns xi_FMR.
        If any uncertainties are provided, also returns xi_FMR_std.
        If debug=True, also returns a dictionary with derivative values
        and variance contributions.
        """

        # Keep track of whether eta was supplied or computed from Vs/Va
        eta_from_ratio = (eta is None)

        if Ms > 1e3:
            Ms = self.mu_0 * Ms

        if eta_from_ratio:
            eta = Vs / Va

        C = (self.e / self.hbar) * (Ms * d_FM * d_NM)
        sqrt_arg = 1 + Meff / Hres
        sqrt_term = np.sqrt(sqrt_arg)
        xi = eta * C * sqrt_term

        xi_var = 0.0
        contributions = {}
        derivatives = {}
        values = {
            "Vs": Vs,
            "Va": Va,
            "eta": eta,
            "Ms": Ms,
            "d_FM": d_FM,
            "d_NM": d_NM,
            "Hres": Hres,
            "Meff": Meff,
            "C": C,
            "sqrt_arg": sqrt_arg,
            "sqrt_term": sqrt_term,
            "xi": xi,
            "eta_from_ratio": eta_from_ratio,
        }

        # Common derivatives
        dxi_dHres = (
            C * eta *
            0.5 * (1 + Meff / Hres) ** (-0.5) *
            (-Meff / Hres**2)
        )
        dxi_dMeff = (
            C * eta *
            0.5 * (1 + Meff / Hres) ** (-0.5) *
            (1 / Hres)
        )

        derivatives["dxi_dHres"] = dxi_dHres
        derivatives["dxi_dMeff"] = dxi_dMeff

        # eta handling
        if eta_from_ratio:
            if Vs_std is not None:
                dxi_dVs = xi / Vs
                derivatives["dxi_dVs"] = dxi_dVs
                contributions["Vs"] = (dxi_dVs * Vs_std) ** 2
                xi_var += contributions["Vs"]

            if Va_std is not None:
                dxi_dVa = -xi / Va
                derivatives["dxi_dVa"] = dxi_dVa
                contributions["Va"] = (dxi_dVa * Va_std) ** 2
                xi_var += contributions["Va"]
        else:
            if eta_std is not None:
                dxi_deta = xi / eta
                derivatives["dxi_deta"] = dxi_deta
                contributions["eta"] = (dxi_deta * eta_std) ** 2
                xi_var += contributions["eta"]

        if Hres_std is not None:
            contributions["Hres"] = (dxi_dHres * Hres_std) ** 2
            xi_var += contributions["Hres"]

        if Meff_std is not None:
            contributions["Meff"] = (dxi_dMeff * Meff_std) ** 2
            xi_var += contributions["Meff"]

        if not contributions:
            if debug:
                return xi, {"values": values, "derivatives": derivatives, "contributions": contributions}
            return xi

        xi_std = np.sqrt(xi_var)

        if debug:
            rel_contributions = {
                k: v / xi_var if xi_var > 0 else np.nan
                for k, v in contributions.items()
            }
            debug_info = {
                "values": values,
                "derivatives": derivatives,
                "contributions": contributions,
                "relative_contributions": rel_contributions,
                "xi_var": xi_var,
                "xi_std": xi_std,
                "relative_error": xi_std / abs(xi) if xi != 0 else np.inf,
            }
            print(debug_info)
            return xi, xi_std

        return xi, xi_std

    def inv_xi_SOT_model(self, x, xi_DL, xi_FL):
        """
        Model for 1/xi_SOT as a function of known quantities.
        x : array-like -> A variable, e.g., thickness t_NM or measured xi_SOT data.
        xi_DL : float -> Damping-like torque efficiency (fit parameter)
        xi_FL : float -> Field-like torque efficiency (fit parameter)
        """
        # Compute the correction factor
        correction = (self.hbar / (self.e * self.mu_0 * self.Ms * self.d_FM * self.d_NM)) * xi_FL

        # Equation: 1/xi_SOT = (1/xi_DL) * (1 + correction)
        return (1 / xi_DL) * (1 + correction)
    
    @staticmethod
    def stfmr_model(H, Vs, Va, Hres, Delta, C, D):
        """
        ST-FMR mixing voltage model.
        """
        # sym = Vs * (Delta**2) / ((H - Hres)**2 + Delta**2)
        # asym = Va * (Delta * (H - Hres)) / ((H - Hres)**2 + Delta**2)
        sym = STFMR.stfmr_symmetric(H, Vs, Hres, Delta, 0, 0)
        asym = STFMR.stfmr_asymmetric(H, Va, Hres, Delta, 0, 0)
        return sym + asym + C + D * H

    def stfmr_symmetric(H, Vs, Hres, Delta, C, D):
        """
        Symmetric part of ST-FMR mixing voltage model.
        """
        return (Vs * Delta**2) / ((H - Hres)**2 + Delta**2) + C + D*H
    
    def stfmr_asymmetric(H, Va, Hres, Delta, C, D):
        """
        Asymmetric part of ST-FMR mixing voltage model.
        """
        return (Va * Delta * (H - Hres)) / ((H - Hres)**2 + Delta**2) + C + D*H

    def fit_stfmr(self, H, V, p0=None, bounds=None, sigma=None):
        """        
        Parameters:
        H : array-like, Field values [mT]
        V : array-like, Mixing voltage [uV]
        p0 : list, optional, Initial guess [Vx, Vy, Hres, Delta, C]
        """
        if p0 is None:
            # crude guesses: peak ~ Vx, min ~ Vy, center ~ Hres
            Delta_guess = 10
            Vs_guess = (max(V))
            # print(Vs_guess, max(V), np.mean(V), min(V))
            Va_guess = (max(V) - min(V)) / 2
            C = np.mean(V)
            D = 0
            if self.used_fields == 'Pos':
                H_min, H_max = 0, np.inf
                # Hres_guess = H.iloc[np.argmax(V)]            
            if self.used_fields == 'Neg':
                H_min, H_max = -np.inf, 0
                # Hres_guess = H.iloc[np.argmin(V)]
            idx = int(np.mean([np.argmax(V[10:-10]), np.argmin(V[10:-10])]))
            Hres_guess = H.iloc[idx]
            print(Hres_guess)
            p0 = [Vs_guess, Va_guess, Hres_guess, Delta_guess, C, D]
            # print(p0)
        # print("Initial guess for fitting parameters:", p0)
        bounds = ( 
            [-500*abs(min(V)), -20000*abs(min(V)), -400, 0, -np.inf, -np.inf], # lower bounds (Delta >= 0) 
            [500*abs(max(V)), 20000*abs(max(V)), 400, 400, np.inf, np.inf] # upper bounds 
            )
        # print(bounds)

        # bounds = ( 
        #     [-np.inf, -np.inf, H_min, abs(H.iloc[1] - H.iloc[0]) , -np.inf], # lower bounds (Delta >= 0) 
        #     [np.inf, np.inf, H_max, np.inf, np.inf] # upper bounds 
        #     )
        # print(p0)
        # print(bounds)

        popt, pcov = curve_fit(
            self.stfmr_model,
            H, V,
            p0=p0,
            bounds=bounds,
            maxfev=100000, 
            sigma=sigma
        )
        print(sigma is None)
        return {
            "Vs": popt[0],
            "Va": popt[1],
            "Hres": popt[2],
            "Delta": popt[3],
            "C": popt[4],
            "D": popt[5]
        }, pcov

    def fit_stfmr2(self, H, V, p0=None, bounds=None, sigma=None):
        """        
        Parameters:
        H : array-like, Field values [mT]
        V : array-like, Mixing voltage [uV]
        p0 : list, optional, Initial guess [Vx, Vy, Hres, Delta, C]
        """
        cutoff = 10
        Hres_guess = H[int(np.mean([np.argmax(V[cutoff:-cutoff]), np.argmin(V[cutoff:-cutoff])]))]

        H_min_data = H.min()
        H_max_data = H.max()

        # --- maximum symmetric half-width ---
        left_width = Hres_guess - H_min_data
        right_width = H_max_data - Hres_guess
        half_width = min(left_width, right_width)
        
        H_low = Hres_guess - half_width
        H_high = Hres_guess + half_width

        mask = (H >= H_low) & (H <= H_high)

        H_fit, V_fit = H[mask], V[mask]

        Delta_guess = (max(H_fit) - min(H_fit)) / 10
        Vs_guess = (max(V_fit))
        # print(Vs_guess, max(V), np.mean(V), min(V))
        Va_guess = (max(V_fit) - min(V_fit)) / 2
        C = 0
        D = 0
        p0 = [Vs_guess, Va_guess, Hres_guess, Delta_guess, C, D]

        # print("Initial guess for fitting parameters:", p0)
        bounds = ( 
            [-500*abs(min(V_fit)), -2000*abs(min(V_fit)), min(H_fit), abs(H_fit.iloc[1] - H_fit.iloc[0]) , -np.inf, -np.inf], # lower bounds (Delta >= 0) 
            [500*abs(max(V_fit)), 2000*abs(max(V_fit)), max(H_fit), max(H_fit), np.inf, np.inf] # upper bounds 
            )
        # print(bounds)

        # bounds = ( 
        #     [-np.inf, -np.inf, H_min, abs(H.iloc[1] - H.iloc[0]) , -np.inf], # lower bounds (Delta >= 0) 
        #     [np.inf, np.inf, H_max, np.inf, np.inf] # upper bounds 
        #     )
        # print(p0)
        # print(bounds)
        popt, pcov = curve_fit(
            self.stfmr_model,
            H, V,
            p0=p0,
            bounds=bounds,
            maxfev=100000, 
            sigma=sigma
        )
        return {
            "Vs": popt[0],
            "Va": popt[1],
            "Hres": popt[2],
            "Delta": popt[3],
            "C": popt[4],
            "D": popt[5]
        }, pcov

class STFMR_thickness_analysis:
    def __init__(self, folder=None, voltage_unit='uV', d_FMs=None, d_NM=None, Ms=None, Width=None, Length=None, used_fields='Pos', group_name='samples'):

        # Physical constants
        self.mu_B = 9.274009994e-24  # Bohr magneton [J/T]
        self.e = 1.602176634e-19     # elementary charge [C]
        self.mu_0 = 4 * np.pi * 1e-7 # vacuum permeability [H/m]
        self.gamma = 1.760859e11     # gyromagnetic ratio [rad/(s·T)]
        self.hbar = 6.626e-34

        # System settings
        self.folder = folder 
        self.voltage_unit = voltage_unit  
        self.d_FMs = d_FMs
        self.d_NM = d_NM
        self.Ms = Ms
        self.Width = Width
        self.Length = Length
        self.used_fields = used_fields
        self.group_name = group_name

    def create_fit_array(self, xmin, xmax, num_points=100):
        if xmin < 0:
            x_fit_min = xmin * 1.1
        if xmax < 0:
            x_fit_max = xmax * 0.9
        else:
            x_fit_min = xmin * 0.9
            x_fit_max = xmax * 1.1
        return np.linspace(x_fit_min, x_fit_max, num_points)

    def analyze_thickness_folder(self):
        results = []

        thickness_map = self.d_FMs
        
        for sample_name in os.listdir(self.folder):
            sample_path = os.path.join(self.folder, sample_name)
            freq_path = os.path.join(sample_path, "FrequencyScan")

            if not os.path.isdir(freq_path):
                # print(f"⚠️ Skipping {sample_name}: no FrequencyScan folder.")
                continue

            # Get thickness from map (or None if missing)
            sample_info = thickness_map.get(sample_name, None)
            if sample_info is None:
                # print(f"⚠️ Thickness for {sample_name} not found.")
                continue
            label = sample_info.get("label", None)

            # print(f"🔍 Analyzing {sample_name} (t_FM= {sample_info['t_FM']} nm)")
            stfmr = STFMR(
                folder=freq_path,
                voltage_unit=self.voltage_unit,
                d_FM=sample_info["t_FM"],
                d_NM=sample_info["t_NM"],
                Ms=self.Ms if sample_info.get("Ms") is None else sample_info["Ms"],
                Width=self.Width,
                Length=self.Length,
                used_fields=self.used_fields
            )

            # try:
            stfmr.calculate_xiFMR(plot=False)
            stfmr.plot_Kittel(plot=False)
            stfmr.plot_linewidth(plot=False)
            result_entry = {
                "Sample": sample_name,
                "t_FM": sample_info.get("t_FM"),
                "t_NM": sample_info.get("t_NM"),
                "Label": label,

                "Xi_fmr": stfmr.xi_fmr,
                "Xi_fmr_std": stfmr.xi_std,
                "Meff": stfmr.Heff_T,
                "Meff_std": stfmr.Heff_T_cov,
                "Alpha": stfmr.alpha,
                "Alpha_std": stfmr.alpha_cov,
                "DeltaH0": stfmr.DeltaH0,
                "DeltaH0_std": stfmr.DeltaH0_cov,

                "freq_": np.array(stfmr.freq_),
                "Vs_": np.array(stfmr.Vs_),
                "Va_": np.array(stfmr.Va_),
                "Hres_": np.array(stfmr.Hres_),
            }

            for key in ["Width", "Length", "R"]:
                if key in sample_info:
                    result_entry[key] = sample_info[key]


            results.append(result_entry)
            # print("Sample:", sample_name, "Vs", np.mean(stfmr.Vs_), "Va", np.mean(stfmr.Va_))
            # print(f"✅ Finished analyzing {sample_name}, xi_FMR= {stfmr.xi_fmr} ± {stfmr.xi_std}")
            # except Exception as e:
            #     print(f"❌ Error analyzing {sample_name}: {e}")
        self.results = pd.DataFrame(results)
        print(self.results)
        # print(self.results["Sample"], self.results["Meff"], self.results["Meff_std"],)
        
        # print(self.results)
        return 
        
    def subtract_mgo_background(self):
        subtracted_results = []

        df = self.results

        for t_fm in df["t_FM"].unique():
            df_t = df[df["t_FM"] == t_fm]

            df_15 = df_t[df_t["Label"] == "15nm"]
            df_mgo = df_t[df_t["Label"] == "Mgo"]

            if df_15.empty or df_mgo.empty:
                continue

            for _, row_15 in df_15.iterrows():
                for _, row_mgo in df_mgo.iterrows():
                    f15 = np.asarray(row_15["freq_"])
                    fmgo = np.asarray(row_mgo["freq_"])

                    common_freqs = np.intersect1d(f15, fmgo)

                    if len(common_freqs) == 0:
                        continue

                    idx_15 = np.where(np.isin(f15, common_freqs))[0]
                    idx_mgo = np.where(np.isin(fmgo, common_freqs))[0]

                    Vs_15 = row_15["Vs_"][idx_15]
                    Va_15 = row_15["Va_"][idx_15]

                    Vs_mgo = row_mgo["Vs_"][idx_mgo]
                    Va_mgo = row_mgo["Va_"][idx_mgo]

                    order = np.argsort(common_freqs)

                    subtracted_results.append({
                        "Sample_15nm": row_15["Sample"],
                        "Sample_mgo": row_mgo["Sample"],
                        "t_FM": t_fm,
                        "t_NM": row_15["t_NM"],
                        "freq": common_freqs[order],
                        "Vs_sub": (Vs_15 - Vs_mgo)[order],
                        "Va_sub": (Va_15 - Va_mgo)[order],
                        "Hres": np.interp(common_freqs, f15, row_15["Hres_"]),
                        "Meff": np.average([row_15["Meff"], row_mgo["Meff"]]),
                    })

 

        self.mu_B = 9.274009994e-24  # Bohr magneton [J/T]
        self.e = 1.602176634e-19     # elementary charge [C]
        self.mu_0 = 4 * np.pi * 1e-7 # vacuum permeability [H/m]
        self.gamma = 1.760859e11     # gyromagnetic ratio [rad/(s·T)]
        self.hbar = 6.626e-34

        df_sub = pd.DataFrame(subtracted_results)

        self.subtracted = df_sub        
        # def FMR_efficiency(Vs, Va, Ms, d_FM, d_NM, Hres, Meff=1):
        # # print('Vs/Va', Vs/Va, 'Hres', (Hres*1e-3)/self.mu_0)
        #     return (Vs/Va) * (self.e/self.hbar) * (self.mu_0 * Ms * d_FM * d_NM ) * np.sqrt(1+(((4* np.pi* Meff))/Hres))
        
        xi_vs_tfm = []


        for _, row in df_sub.iterrows():

            Vs = row["Vs_sub"]
            Va = row["Va_sub"]
            Hres = row["Hres"]
            t_FM = row["t_FM"]
            t_NM = row["t_NM"]
            Meff = row["Meff"]

            # Safety mask
            mask = np.isfinite(Vs) & np.isfinite(Va) & (Va != 0)

            Vs = Vs[mask]
            Va = Va[mask]
            Hres = Hres[mask]

            xi = FMR_efficiency(
                Vs=Vs,
                Va=Va,
                Ms=self.Ms,
                d_FM=abs(t_FM),
                d_NM=abs(t_NM),
                Hres=Hres,
                Meff=Meff
            )

            # Average over frequency (simple or weighted)
            xi_avg = np.mean(xi)
            xi_std = np.std(xi)

            xi_vs_tfm.append({
                "t_FM": t_FM,
                "xi_FMR": xi_avg,
                "xi_FMR_err": xi_std
            })
        df_xi = pd.DataFrame(xi_vs_tfm).sort_values("t_FM")
        for col in df_xi.columns:
            print(f"{col} = {df_xi[col].tolist()}\n")        
        plt.figure(figsize=(8,5))
        plt.errorbar(
            df_xi["t_FM"],
            df_xi["xi_FMR"],
            yerr=df_xi["xi_FMR_err"],
            fmt='o',
            capsize=4
        )

        plt.xlabel("t_FM (nm)")
        plt.ylabel(r"$\xi_{\mathrm{FMR}}$")
        plt.title(r"Background-subtracted FMR efficiency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return subtracted_results
    
    def subtract_Mgo_background_old(self):
        BASE_DIR = os.getcwd()
        thickness_map = self.d_FMs
        # -----------------------------
        # STEP 1: ENSURE AVERAGED DATA EXISTS
        # -----------------------------
        for sample, info in thickness_map.items():
            freq_path = os.path.join(BASE_DIR, sample, "FrequencyScan")
            avg_path = os.path.join(freq_path, "Averaged_Traces_posB")

            if not os.path.isdir(avg_path):
                print(f"[INFO] Running STFMR for {sample}")

                stfmr = STFMR(
                    folder=freq_path,
                    voltage_unit=self.voltage_unit,
                    d_FM=info["t_FM"],
                    d_NM=info["t_NM"],
                    Ms=self.Ms,
                    Width=self.Width,
                    Length=self.Length,
                    used_fields=self.used_fields
                )
                stfmr.analyze_traces()
            else:
                print(f"[OK] Averaged data exists for {sample}")

        by_thickness = defaultdict(dict)

        for sample, info in thickness_map.items():
            t_fm_nm = info["t_FM"] * 1e9  # convert to nm
            by_thickness[t_fm_nm][info["label"]] = sample

        def subtract_vx(file_15nm, file_mgo):
            """
            Returns dataframe where Vx = Vx_MgO - Vx_15nm
            """
            df_15 = pd.read_csv(file_15nm, skiprows=1, delim_whitespace=True, comment="#")
            df_mgo = pd.read_csv(file_mgo, skiprows=1, delim_whitespace=True, comment="#")

            df_out = df_mgo.copy()
            df_out["Vx"] = df_mgo["Vx"] - df_15["Vx"]

            return df_out

        for t_fm_nm, samples in by_thickness.items():

            if "15nm" not in samples or "Mgo" not in samples:
                print(f"[SKIP] Missing pair for t_FM = {t_fm_nm} nm")
                continue

            s_15 = samples["15nm"]
            s_mg = samples["Mgo"]

            path_15 = os.path.join(BASE_DIR, s_15, "FrequencyScan", "Averaged_Traces_posB")
            path_mg = os.path.join(BASE_DIR, s_mg, "FrequencyScan", "Averaged_Traces_posB")

            out_dir = os.path.join(
                BASE_DIR,
                f"Subtracted_t_fm_{t_fm_nm:.1f}nm"
            )
            os.makedirs(out_dir, exist_ok=True)

            files_15 = sorted(os.listdir(path_15))

            for fname in files_15:
                f15 = os.path.join(path_15, fname)
                fmg = os.path.join(path_mg, fname)

                if not os.path.isfile(fmg):
                    print(f"[WARN] Missing MgO file: {fname}")
                    continue

                df_sub = subtract_vx(f15, fmg)

                out_file = os.path.join(out_dir, fname)
                df_sub.to_csv(out_file, sep="\t", index=False)

            print(f"[DONE] Subtraction complete for t_FM = {t_fm_nm:.1f} nm")


    def plot_xi(self):
        # Convert thickness to nm
        t_nm = self.results['t_FM'] * 1e9
        x = t_nm

        y = self.results['Xi_fmr']
        yerr = self.results['Xi_fmr_std'] 

        # Boolean masks
        mask_pos = t_nm > 0
        mask_neg = t_nm < 0

        plt.figure(figsize=(8,5))

        # Positive thickness (blue)
        plt.errorbar(x[mask_pos], y[mask_pos], yerr=yerr[mask_pos],
                    fmt='o', color='blue', label='Power 0 dBm')

        # Negative thickness (red)
        plt.errorbar(abs(x[mask_neg]), y[mask_neg], yerr=yerr[mask_neg],
                    fmt='o', color='red', label='Power 5 dBm')

        # plt.ylim(-1, 1)
        
        plt.xlabel('t_FM (nm)')
        plt.ylabel('$\\xi_{FMR}$')
        plt.title(f'$\\xi_{{FMR}}$ for {self.group_name}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_xi_vs_tNM(self):
        # Convert thickness to nm
        t_nm = self.results['t_NM'] * 1e9
        x = t_nm
        y = self.results['Xi_fmr']
        yerr = self.results['Xi_fmr_std'] 
        print(x, y)

        def linear(x, a, b):
            return a * x + b
        popt, pcov = curve_fit(linear, x, y)
        x_fit = self.create_fit_array(min(x), max(x))
        plt.figure(figsize=(6,4))
        plt.errorbar(x, y, yerr=yerr,
                    fmt='o', color='blue', label='CoFeB(4)/Pt(t)')
        plt.plot(x_fit, linear(x_fit, *popt), '--', label=f'Linear fit: slope={popt[0]:.2g}, intercept={popt[1]:.2g}')
        plt.xlim(left=0)
        # plt.ylim(bottom=0)

        plt.xlabel('$t_{Pt}$ (nm)', fontsize=12)
        plt.ylabel('$\\xi_{FMR}$', fontsize=12)
        # plt.title(f'$\\xi_{{FMR}}$ for {self.group_name}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
                # Convert thickness to nm
        t_nm = self.results['t_NM'] * 1e9
        x = t_nm

        y = self.results['Meff']
        yerr = self.results['Meff_std'] 

        def linear(x, a, b):
            return a * x + b
        popt, pcov = curve_fit(linear, x, y)
        x_fit = self.create_fit_array(min(x), max(x))
        plt.figure(figsize=(8,5))
        plt.errorbar(x, y, yerr=yerr,
                    fmt='o', color='blue', label='CoFeB(4)/Pt(t)')
        plt.plot(x_fit, linear(x_fit, *popt), '--', label=f'Linear fit: slope={popt[0]:.3e}, intercept={popt[1]:.3e}')
        plt.xlim(left=0)
        plt.xlabel('t_NM (nm)')
        plt.ylabel('$\\xi_{FMR}$')
        plt.title(f'$\\xi_{{FMR}}$ for {self.group_name}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_xi_inv(self, fit=True):

        fig, ax = plt.subplots(figsize=(6, 4))   # create figure ONCE

        t_nm = self.results['t_FM'] * 1e9
        err_invs = self.results['Xi_fmr_std']/(self.results['Xi_fmr']**2)

        if err_invs.isna().any():
            print("Warning: err_invs contains NaNs. Replacing them with 2×max value.")
        err_invs = err_invs.fillna(err_invs.max())

        xdata = 1/(self.results['t_FM']*1e9)
        ydata = 1/self.results['Xi_fmr']

        ax.errorbar(xdata, ydata, yerr=err_invs, fmt='o', label='Data', color='black')

        def f(x, a, b):
            return a * x + b

        if fit:
            popt, pcov = curve_fit(f, xdata, ydata, p0=[0.1, 0.1], sigma=err_invs)

            a, b = popt
            sigma_a, sigma_b = np.sqrt(np.diag(pcov))

            xi_DL = 1 / b
            xi_DL_err = sigma_b / b**2

            xi_FL = a * self.e * self.mu_0 * self.Ms * self.d_NM / self.hbar
            xi_FL_err = sigma_a * self.e * self.mu_0 * self.Ms * self.d_NM / self.hbar

            print(f"Fitted xi_DL = {xi_DL:.3e} ± {xi_DL_err:.3e}")
            print(f"Fitted xi_FL = {xi_FL:.3e} ± {xi_FL_err:.3e}")

            t_fit = self.create_fit_array(0, max(xdata))

            ax.plot(t_fit, f(t_fit, a, b), '--',
                    label=f'Fit slope={a:.1f}, intercept={b:.2g}')

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=-100, top=30)
        ax.set_xlabel('1 / $t_{CoFeB}$ ($nm^{-1}$)', fontsize=12)
        ax.set_ylabel('1 / $\\xi_{FMR}$', fontsize=12)
        # ax.set_title(f'Fit of $1/\\xi_{{FMR}}$ vs $1/t_{{CoFeB}}$ with {self.group_name}', fontsize=14)
        ax.grid(True)
        ax.legend()

        plt.show()

    def plot_xi_inv_grouped(
        self,
        group_col="Label",
        fit=True,
        legend_loc="best"
    ):
        """
        Plot and fit 1/xi_FMR vs 1/t_FM for multiple sample groups.
        """

        df = self.results.copy()

        fig, ax = plt.subplots(figsize=(8, 5))

        def f(x, a, b):
            return a * x + b

        for label, g in df.groupby(group_col):

            # --- x and y data ---
            t_nm = g["t_FM"] * 1e9
            xdata = 1 / t_nm
            ydata = 1 / g["Xi_fmr"]

            # --- error propagation ---
            err_invs = g["Xi_fmr_std"] / (g["Xi_fmr"] ** 2)
            if err_invs.isna().any():
                err_invs = err_invs.fillna(err_invs.max())

            # --- plot data ---
            ax.errorbar(
                xdata,
                ydata,
                yerr=err_invs,
                fmt="o",
                label=f"{label} data"
            )

            # --- fit ---
            if fit and len(g) >= 2:
                popt, pcov = curve_fit(
                    f,
                    xdata,
                    ydata,
                    p0=[0.1, 0.1],
                    sigma=err_invs,
                    absolute_sigma=True
                )

                a, b = popt
                sigma_a, sigma_b = np.sqrt(np.diag(pcov))

                xi_DL = 1 / b
                xi_DL_err = sigma_b / b**2

                print(
                    f"[{label}] "
                    f"xi_DL = {xi_DL:.3e} ± {xi_DL_err:.3e}, "
                    f"xi_FL = {a:.3e} ± {sigma_a:.3e}"
                )

                # --- plot fit line ---
                x_fit = np.linspace(0, xdata.max() * 1.05, 200)
                ax.plot(
                    x_fit,
                    f(x_fit, a, b),
                    "--",
                    label=f"{label} fit"
                    , color='red'
                )

        # --- formatting ---
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=30)
        ax.set_xlabel(r"$1/t_{CoFeB}$ (nm$^{-1}$)")
        ax.set_ylabel(r"$1/\xi_{FMR}$")
        ax.set_title(r"Grouped fit of $1/\xi_{FMR}$ vs $1/t_{CoFeB}$")
        ax.grid(True)
        ax.legend(loc=legend_loc)
        plt.tight_layout()
        plt.show()


    def plot_linewidths(self):
        thickness_map = self.d_FMs

        fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
        color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        for sample_name, sample_info in sorted(thickness_map.items(), key=lambda x: x[1]["t_FM"]):
            sample_path = os.path.join(self.folder, sample_name)
            freq_path = os.path.join(sample_path, "FrequencyScan")

            if not os.path.isdir(freq_path):
                print(f"⚠️ Skipping {sample_name}: no FrequencyScan folder.")
                continue

            print(f"🔍 Analyzing {sample_name} (t_FM= {sample_info['t_FM']} nm)")

            stfmr = STFMR(
                folder=freq_path,
                voltage_unit=self.voltage_unit,
                d_FM=sample_info["t_FM"],
                d_NM=sample_info["t_NM"],
                Ms=self.Ms,
                Width=self.Width,
                Length=self.Length,
                used_fields=self.used_fields
            )

            try:
                df = stfmr.plot_linewidth(plot=False)

                color = next(color_cycle)

                freq = np.linspace(0, max(df["freq"]) * 1.1, 200)

                label = (
                    rf"$t_{{CoFeB}}$ = {sample_info['t_FM'] * 1e9:.1f}$\,\mathrm{{nm}}$ "
                    # rf"$\Delta H_0$ = {stfmr.DeltaH0 * self.mu_0 * 1e3:.1f}$\,\mathrm{{mT}}$, "
                    # rf"$\alpha$ = {stfmr.alpha:.2g}"
                )

                # --- Data points ---
                axes.errorbar(
                    df["freq"],
                    df["Delta"],
                    yerr=df["Deltastd"],
                    fmt="o",
                    color=color,
                    label="_nolegend_"
                )

                # --- Fit curve ---
                axes.plot(
                    freq,
                    stfmr.linewidth_formula(freq * 1e9, stfmr.DeltaH0, stfmr.alpha) * 1e3,
                    "--",
                    color=color,
                    label=label
                )

                axes.set_xlabel("Frequency (GHz)", fontsize=12)
                axes.set_ylabel(r"$\Delta$ (mT)", fontsize=12)
                axes.grid(True)
                axes.legend(fontsize=14, loc="upper left")
                axes.set_xlim(left=0, right=18)
                # axes.set_ylim(bottom=0, top=10)

                # plt.suptitle("Linewidth versus frequency", fontsize=14)
                plt.tight_layout()

            except Exception as e:
                print(f"❌ Error analyzing {sample_name}: {e}")

        plt.show()


    def plot_frequency(self, freq=8):
        thickness_map = self.d_FMs
        results = []

        fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharex=True)
        color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        for sample_name, sample_info in sorted(thickness_map.items(),key=lambda x: x[1]["t_FM"]):            
            color = next(color_cycle)
            sample_path = os.path.join(self.folder, sample_name)
            freq_path = os.path.join(sample_path, "FrequencyScan")
            if not os.path.isdir(freq_path):
                print(f"⚠️ Skipping {sample_name}: no FrequencyScan folder.")
                continue

            # Get thickness from map (or None if missing)
            print(f"🔍 Analyzing {sample_name} (t_FM= {sample_info['t_FM']} nm)")

            stfmr = STFMR(
                folder=freq_path,
                voltage_unit=self.voltage_unit,
                d_FM=sample_info["t_FM"],
                d_NM=sample_info["t_NM"],
                Ms=self.Ms,
                Width=self.Width,
                Length=self.Length,
                used_fields=self.used_fields
            )
            try: 
                for file in os.listdir(freq_path):
                    if file.lower().endswith(".txt"):
                        key = stfmr.get_key(file)
                        if float(key[1]) == freq:
                            full_file_path = os.path.join(freq_path, file)
                            R, df = stfmr.read_file(full_file_path)
                            plt.plot(df["H_given"], df["Vx"], '.', label=f"Sample {sample_name} with t_FM= {sample_info['t_FM']:.2g} nm", color=color)
                            break
            except Exception as e:
                print(f"❌ Error analyzing {sample_name}: {e}")
        plt.xlabel("Magnetic Field (mT)")
        plt.ylabel(f"Mixing Voltage ({self.voltage_unit})")
        plt.title(f"ST-FMR at {freq} GHz for {self.group_name}")
        plt.legend()
        plt.show()

    def plot_average_frequency(self, freq=8, fit=False):
        if self.used_fields== 'Pos':
            average_folder = r"Averaged_Traces_posB"
        elif self.used_fields== 'Neg':
            average_folder =  r"Averaged_Traces_negB"
        thickness_map = self.d_FMs
        results = []

        fig, ax = plt.subplots(1, 1, figsize=(12, 10), sharex=True)
        # color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        for sample_name, sample_info in sorted(thickness_map.items(),key=lambda x: x[1]["t_FM"]):            
            # color = next(color_cycle)
            sample_path = os.path.join(self.folder, sample_name)
            freq_path = os.path.join(sample_path, "FrequencyScan")
            folder_path = os.path.join(freq_path, average_folder)
            # Get thickness from map (or None if missing)
            print(f"🔍 Analyzing {sample_name} (t_FM= {sample_info["t_FM"]} nm)")

            stfmr = STFMR(
                folder=freq_path,
                voltage_unit=self.voltage_unit,
                d_FM=sample_info["t_FM"],
                d_NM=sample_info["t_NM"],
                Ms=self.Ms,
                Width=self.Width,
                Length=self.Length,
                used_fields=self.used_fields
            )
            if not os.path.exists(folder_path):
                stfmr.analyze_traces()
                continue
            try: 
                for file in os.listdir(folder_path):
                    if file.lower().endswith(".txt"):
                        key = stfmr.get_key(file)
                        if float(key[1]) == freq:
                            full_file_path = os.path.join(folder_path, file)

                            R, df = stfmr.read_file(full_file_path)
                            # print(file)
                            # plt.errorbar(df["H_given"], df["Vx"], df['Vx_err'], label=f"Sample {sample_name} with t_FM= {thickness:.2g} nm", color=color)
                            if fit:
                                fit_params, pcov = stfmr.fit_Vmix(full_file_path, plot=False)
                            if self.voltage_unit == 'uV':
                                df['Vx'] *= 1e6  # Convert Vx to microvolts
                                df['Vy'] *= 1e6  # Convert Vy to microvolts
                                df['Vx_err'] *= 1e6
                                df['Vy_err'] *= 1e6
                            stfmr.plot_Vmix(
                                df,
                                R,
                                fit_params=fit_params if fit else None,
                                ax=ax,
                                sample=sample_name
                            )
                            break
            except Exception as e:
                print(f"❌ Error analyzing {sample_name}: {e}")
        ax.set_xlabel("Magnetic Field (mT)")
        ax.set_ylabel(f"Mixing Voltage ({self.voltage_unit})")
        ax.set_title(f"ST-FMR at {freq} GHz")
        ax.legend()
        plt.show()

    def plot_kittels(self):
        thickness_map = self.d_FMs

        fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
        color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        for sample_name, sample_info in sorted(thickness_map.items(),key=lambda x: x[1]["t_FM"]):            
            sample_path = os.path.join(self.folder, sample_name)
            freq_path = os.path.join(sample_path, "FrequencyScan")

            if not os.path.isdir(freq_path):
                print(f"⚠️ Skipping {sample_name}: no FrequencyScan folder.")
                continue

            # Get thickness from map (or None if missing)
            print(f"🔍 Analyzing {sample_name} (t_FM= {sample_info['t_FM']} nm)")

            stfmr = STFMR(
                folder=freq_path,
                voltage_unit=self.voltage_unit,
                d_FM=sample_info["t_FM"],
                d_NM=sample_info["t_NM"],
                Ms=self.Ms,
                Width=self.Width,
                Length=self.Length,
                used_fields=self.used_fields
            )

            try:
                df = stfmr.plot_Kittel2(plot=False)  # Fit without showing individual plots
                mask = (df['freq'] >= 10) & (df['freq'] <= 16)

                color = next(color_cycle)

                if self.used_fields == 'Pos':
                    H = np.linspace(0, max(df['Hres'][mask]) * 1.1, 200)
                elif self.used_fields == 'Neg':
                    H = np.linspace(1.1 * min(df['Hres'][mask]), 0, 200)
                label = rf"$t_{{CoFeB}}$ = {sample_info['t_FM'] * 1e9:.1f}" #, $H_{{k}}$ = {stfmr.Hk * 1e3:.1f}$\,\mathrm{{mT}}$ \,\mathrm{{nm}}$, $\mu_0 M_{{eff}}$ = {stfmr.Heff_T * 1e3:.1f}$\,\mathrm{{mT}}$

                # Data points
                axes.errorbar(
                    df['Hres'][mask], df['freq'][mask],
                    0, df['Hresstd'][mask],
                    "o",
                    color=color,
                    label="_nolegend_"   # hide from legend
                )

                # Fit curve
                axes.plot(
                    H,
                    stfmr.Kittel_formula2(abs(H) * 1e-3, stfmr.Heff_T, stfmr.Hk),
                    '--',
                    color=color,
                    label=label
                )
                axes.set_xlabel("$H_{res}$ (mT)", fontsize=12)
                axes.set_ylabel("Frequency (GHz)", fontsize=12)
                axes.grid(True)
                axes.legend(fontsize=14)
                # axes.set_ylim(bottom=0, top=30)

                # plt.suptitle(f"Radio frequency versus resonance field", fontsize=14)
                plt.tight_layout()

            except Exception as e:
                print(f"❌ Error analyzing {sample_name}: {e}")
        plt.show()

            
    def plot_DeltaH0(self):
        # Boolean masks
        t_nm = self.results['t_FM'] * 1e9
        mask_pos = t_nm > 0
        mask_neg = t_nm < 0

        plt.figure(figsize=(8,5))
        plt.errorbar(t_nm[mask_pos], self.results['DeltaH0'][mask_pos]*1e-3, yerr=self.results['DeltaH0_std'][mask_pos]*1e-3, fmt='o', label='Pow0 samples', color='blue')
        plt.errorbar(t_nm[mask_neg]*-1, self.results['DeltaH0'][mask_neg]*1e-3, yerr=self.results['DeltaH0_std'][mask_neg]*1e-3, fmt='o', label='Pow5 samples', color='red')
        plt.xlabel('(t_FM (nm))')
        plt.ylabel('Inhomogeneous Broadening $\\mu_0 \\Delta H$ (mT)')
        plt.ylim(bottom=0)
        plt.title('DeltaH0 vs Ferromagnet Thickness')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_Meff(self, data=None):
        if data is None:
            data = self.results
        df = pd.DataFrame(data)
        df = df.reset_index()

        print("Meff", df['Meff'], df["Sample"])
        # Convert thickness to nm and T to mT
        x = df['t_NM'] * 1e9
        y = df['Meff'] 
        yerr = df['Meff_std'] 

        plt.figure(figsize=(6,4))
        # Positive thickness (blue)
        plt.errorbar(x, y, yerr=yerr,
                    fmt='o', color='blue', label=f'{df["Sample"].iloc[0]}')
        plt.xlabel(r'$t_{Pt}$ (nm)', fontsize=12)
        plt.ylabel(r'$H_{eff}$ (T)', fontsize=12)
        plt.ylim(bottom=0, top=1.400)
        plt.xlim(left=0)
        # plt.title('Fit of Meff vs Ferromagnet Thickness')
        plt.grid(True)
        # plt.legend()
        plt.show()
        
        mu0 = self.mu_0
        fig, ax = plt.subplots(figsize=(6,4))

        results = []

        color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        for label, g in df.groupby("Label"):

            color = next(color_cycle)

            g = g.sort_values("t_FM")

            x = (1/(g["t_FM"].astype(float)*1e9)).to_numpy()
            y = g["Meff"].astype(float).to_numpy()
            yerr = g["Meff_std"].astype(float).to_numpy()

            # avoid zero weights
            yerr[yerr <= 0] = np.min(yerr[yerr > 0])

            weights = 1/yerr

            # weighted linear regression
            p, cov = np.polyfit(x, y, 1, w=weights, cov=True)

            b, a = p
            b_err, a_err = np.sqrt(np.diag(cov))

            # physical parameters
            Ms = a / mu0
            Ms_err = a_err / mu0

            Ks = -b * Ms / 2
            Ks_err = np.sqrt(
                (Ms*b_err/2)**2 +
                (b*Ms_err/2)**2
            )

            results.append({
                "Label": label,
                "Ms": Ms,
                "Ms_err": Ms_err,
                "Ks": Ks,
                "Ks_err": Ks_err
            })
            results_df = pd.DataFrame(results)
            # plot data
            ax.errorbar(
                x, y, yerr=yerr,
                fmt="o",
                color=color,
                # label=label
            )

            # plot fit
            x_fit = np.linspace(min(x), max(x), 200)
            y_fit = a + b*x_fit

            ax.plot(
                x_fit,
                y_fit,
                "--",
                color=color,
                label=f'{label} fit: Ks={Ks:.2g} J/m², Ms={Ms:.2g} A/m'
            )

        ax.set_xlabel(r"$1/t_{CoFeB}$ (nm$^{-1}$)", fontsize=14)
        ax.set_ylabel(r"$H_{\mathrm{eff}}$ (T)", fontsize=14)
        ax.grid(True)
        ax.legend()

    def inv_xi_model(self, t_FM, xi_FL, xi_DL):
        return (1/xi_DL) * (1 + (self.hbar * xi_FL / self.e * self.mu_0 * self.Ms * t_FM * self.d_NM))

    def plot_grouped_errorbars(
        self,
        df,
        x,
        y,
        yerr=None,
        label_col="Label",
        xlabel="",
        ylabel="",
        title=None,
        scale_x=1.0,
        legend_loc="best"
    ):
        """
        Generic grouped errorbar plot.
        Each unique value in label_col is plotted as a separate group.
        """
        # print(df[x], df[y], df[yerr] if yerr else "No yerr")
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        for label, g in df.groupby(label_col):
            ax.errorbar(
                g[x] * scale_x,
                g[y],
                yerr=g[yerr] if yerr is not None and yerr in g else None,
                fmt="o",
                label=str(label),
                markersize=5
            )
        ax.set_xlim(left=0)
        # ax.set_ylim(bottom=0)

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        # if title:
        #     ax.set_title(title)
        ax.grid(True)
        plt.legend()
        # plt.tight_layout()
        plt.show()


    def compute_means_from_arrays(self, df, column):
        return df[column].apply(
            lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else np.nan
        )

    def compute_stds_from_arrays(self, df, column):
        return df[column].apply(
            lambda x: np.std(x, ddof=1) if isinstance(x, (list, np.ndarray)) else np.nan
        )

    def plot_different_groups(self, title=None):
        df = self.results.copy()
        df["t_FM_inv"] = 1/df["t_FM"]        
        print(df.head())
        # ---- xi_FMR vs t_NM ----
        self.plot_grouped_errorbars(
            df=df,
            x="t_FM",
            y="Xi_fmr",
            yerr="Xi_fmr_std",
            scale_x=1e9,
            xlabel="$t_{CoFeB}$ (nm)",
            ylabel=r"$\xi_{FMR}$",
            legend_loc="lower left",
            title=title
        )
        
        self.plot_grouped_errorbars(
            df=df,
            x="t_FM",
            y="Alpha",
            yerr="Alpha_std",
            scale_x=1e9,
            xlabel="$t_{CoFeB}$ (nm)",
            ylabel=r"$\alpha$",
            legend_loc="lower left",
            title=title
        )

        # ---- Meff vs t_FM ----
        self.plot_grouped_errorbars(
            df=df,
            x="t_FM",
            y="Meff",
            yerr="Meff_std",
            scale_x=1e9,
            xlabel="t_FM (nm)",
            ylabel=r"$M_{\mathrm{eff}}$",
            legend_loc="lower left"
        )

        # # # ---- Compute averages ----
        df["Vs_mean"] = self.compute_means_from_arrays(df, "Vs_")
        df["Vs_std"]  = self.compute_stds_from_arrays(df, "Vs_")
        df["Va_mean"] = self.compute_means_from_arrays(df, "Va_")
        df["Va_std"]  = self.compute_stds_from_arrays(df, "Va_")

        # # # ---- ⟨Vs⟩ ----
        self.plot_grouped_errorbars(
            df=df,
            x="t_FM",
            y="Vs_mean",
            yerr="Vs_std",
            scale_x=1e9,
            xlabel="$t_{CoFeB}$ (nm)",
            ylabel=r"S ($\mu V$)"
        )

        # # ---- ⟨Va⟩ ----
        self.plot_grouped_errorbars(
            df=df,
            x="t_FM",
            y="Va_mean",
            yerr="Va_std",
            scale_x=1e9,
            xlabel="$t_{CoFeB}$ (nm)",
            ylabel=r"A ($\mu V$)"
        )

    def dimension_analysis(self):
        df = self.results.copy()
        print(df.head())
        title="Dependence of the FMR efficiency with device dimensions"

        fig, ax = plt.subplots(3, 1, figsize=(10, 14))

        for label, g in df.groupby("Length"):
            ax[0].errorbar(
                g["Width"] * 1e6,
                g["Xi_fmr"],
                yerr=g["Xi_fmr_std"],
                fmt="o",
                label=f"Length = {label*1e6:.2g} $\\mu$m"
            )
        ax[0].set_xlabel("Width ($\\mu$m)", fontsize=12)
        ax[0].set_ylabel("$\\xi_{FMR}$", fontsize=12)
        ax[0].set_title(title, fontsize=14)
        ax[0].grid(True)
        ax[0].legend(loc="lower right")

        df["aspectratio"] = df["Length"]/df["Width"]
        print(df["aspectratio"], df)
        ax[1].errorbar(
            df["aspectratio"],
            df["Xi_fmr"],
            yerr=df["Xi_fmr_std"],
            fmt="o",
            label=f"Length/Width"
        )
        ax[1].set_xlabel("Aspectratio", fontsize=12)
        ax[1].set_ylabel("$\\xi_{FMR}$", fontsize=12)
        ax[1].grid(True)
        ax[1].legend(loc="lower right")


        ax[2].errorbar(
            df["R"],
            df["Xi_fmr"],
            yerr=df["Xi_fmr_std"],
            fmt="o",
        )
        ax[2].set_xlabel("Resistance (Ohm)", fontsize=12)
        ax[2].set_ylabel("$\\xi_{FMR}$", fontsize=12)
        ax[2].grid(True)
        plt.legend()
        plt.show()




if __name__ == "__main__":
    stfmr = STFMR(folder=r"Path\to\your\datafolder", d_FM=6e-9, d_NM=6e-9, Ms=800e3, Width=20e-6, Length=100e-6, used_fields='Pos')
    print(stfmr.calculate_current_density(power_dbm=15))

