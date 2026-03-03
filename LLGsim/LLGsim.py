import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

class LLGsim:
    def __init__(self, Hext=np.array([0, 0, 1]), m0 = np.array([1.0, 0, 0.0]), sigma=None, Ms=1, alpha=0, gamma=1.76e11, H_drive_amp=10):
        # physical constants
        self.mu0 = 4.0 * np.pi * 1e-7
        self.gamma = gamma  # rad/(s·T)
        self.alpha = alpha
        self.f_drive = 2.8e9

        # simulation parameters
        self.dt = 1e-14
        self.n_steps = 1000
        self.t = np.arange(0, self.n_steps) * self.dt
    
        self.Hext = Hext
        self.m0 = m0/np.linalg.norm(m0)
        self.Ms = 1

        # torques or fields
        self.sigma = sigma
        self.tau_DL_amp = 5e10
        self.tau_FL_amp = 0
        self.H_drive_amp = H_drive_amp

    def H_drive(self, t):
        Hx = self.H_drive_amp * np.cos(2 * np.pi * self.f_drive * t)
        Hy = self.H_drive_amp * np.sin(2 * np.pi * self.f_drive * t)
        Hz = 0
        return np.array([Hx, Hy, Hz])

    def sim_m_vector(self):
        m_steps = np.zeros((self.n_steps, 3))
        m_steps[0] = self.m0
        for i in range(1, self.n_steps):
            m_steps[i] = self.rk4_step(m_steps[i-1], i)
        self.m_steps = m_steps

    def animate_m_vector(self):
        # Creating figure
        fig, ax = self.plot_m_vector()

        # dynamic vector and trace
        self.vector, = ax.plot([], [], [], 'o-', color='blue', lw=3)
        self.trace, = ax.plot([], [], [], color='cyan', alpha=0.6)

        # Create animation
        ani = FuncAnimation(
            fig, self.update_animation,
            frames=range(0, len(self.m_steps), 100),  # every 100th frame
            init_func=self.initialize_animation,
            interval=20,
            blit=True
        )

        # Save
        ani.save("llg_precession.gif", writer='pillow', fps=10)
        return
    
    def initialize_animation(self):
        self.vector.set_data([], [])
        self.vector.set_3d_properties([])
        self.trace.set_data([], [])
        self.trace.set_3d_properties([])
        return self.vector, self.trace
    
    def update_animation(self, frame):
        mx, my, mz = self.m_steps[frame]
        self.vector.set_data([0, mx], [0, my])
        self.vector.set_3d_properties([0, mz])
        self.trace.set_data(self.m_steps[:frame, 0], self.m_steps[:frame, 1])
        self.trace.set_3d_properties(self.m_steps[:frame, 2])
        return self.vector, self.trace
    
    def plot_m_vector(self):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
        ax.set_xlabel('m_x'); ax.set_ylabel('m_y'); ax.set_zlabel('m_z')
        ax.set_title('LLG Magnetization Precession')

        # draw unit sphere for context
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        # ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                        # color='lightgray', alpha=0.3)
        ax.grid(False)

        # draw external field arrow
        ax.quiver(0, 0, 0, *self.Hext, color='red', length=0.8, normalize=True)
        ax.quiver(0, 0, 0, *self.m0, color='blue', length=0.8, normalize=True)

        return fig, ax
    
    def plot_m_components(self):
        fig = plt.figure(figsize=(6,6))
        plt.ylim([-1, 1])
        plt.xlim([0, 1e-9])
        plt.title('LLG Magnetization')
        plt.plot(self.t, self.m_steps[:,0], label='mx')
        plt.plot(self.t, self.m_steps[:,1], label='my')
        plt.plot(self.t, self.m_steps[:,2], label='mz')
        plt.legend()
        plt.show()
        return 

    def llg_dm_dt(self, m, H_eff, i):
        pre = self.gamma / (1 + self.alpha**2)
        mxh = np.cross(m, H_eff)
        # print('mxh', mxh)
        mxmxh = np.cross(m, mxh)
        if self.sigma is not None:
            torques = self.DLT_and_FLT(m, i)
            # print(torques)
        else:
            torques = 0
        return -pre * mxh - (pre * self.alpha * mxmxh)/self.Ms + torques

    def effective_field(self, m, t):
        return self.Hext + self.H_drive(t)
    
    def DLT_and_FLT(self, m, i):
        tau_dl = self.tau_DL_amp * np.cos(2 * np.pi * self.f_drive * i * self.dt)
        tau_fl = self.tau_FL_amp * np.cos(2 * np.pi * self.f_drive * i * self.dt)
        DLT = tau_dl * np.cross(m, np.cross(m, self.sigma))   # damping-like torque
        FLT = tau_fl * np.cross(m, self.sigma) 
        return DLT + FLT

    def m_components(self, m, theta, phi):
        mx = m * np.sin(theta) * np.cos(phi)
        my = m * np.sin(theta) * np.sin(phi)
        mz = m * np.cos(theta) 
        return mx, my, mz
    
    def rk4_step(self, m, i):
        H = self.effective_field(m, i * self.dt)
        k1 = self.llg_dm_dt(m, H, i)
        k2 = self.llg_dm_dt(m + 0.5*self.dt*k1, H, i)
        k3 = self.llg_dm_dt(m + 0.5*self.dt*k2, H, i)
        k4 = self.llg_dm_dt(m + self.dt*k3, H, i)
        m_new = m + (self.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        return m_new / np.linalg.norm(m_new)