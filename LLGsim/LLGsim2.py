import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import re
import os 
import glob
import sys
from pathlib import Path
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

class LLGsim:
    def __init__(self, Hext=None, m0=None, Ms=1, alpha=0.05, gamma=1.76e11, dt=1e-12, n_steps=5000):
        # physical constants
        self.mu0 = 4.0 * np.pi * 1e-7
        self.gamma = gamma  # rad/(s·T)
        self.alpha = alpha

        # simulation parameters
        self.dt = dt
        self.n_steps = n_steps
        self.t = np.arange(0, self.n_steps) * self.dt
    
        self.Hext = Hext
        self.m0 = m0/np.linalg.norm(m0)
        self.Ms = 1

    def sim_m_vector(self):
        m_steps = np.zeros((self.n_steps, 3))
        m_steps[0] = self.m0
        for i in range(1, self.n_steps):
            m_steps[i] = self.rk4_step(m_steps[i-1], i)
        self.m_steps = m_steps

    def animate_m_vector(self):
        # Create figure with two subplots
        fig = plt.figure(figsize=(15,15))
        ax3d = fig.add_subplot(projection='3d',)   # 3D vector subplot
        # ax3d.set_facecolor('none')
        # fig.patch.set_alpha(0)

        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        # Optional: remove pane edges
        ax3d.xaxis.pane.set_edgecolor('none')
        ax3d.yaxis.pane.set_edgecolor('none')
        ax3d.zaxis.pane.set_edgecolor('none')
        # Setup 3D subplot
        ax3d.set_xlim([-1,1]); ax3d.set_ylim([-1,1]); ax3d.set_zlim([-0.1,1])
        ax3d.set_xlabel('m_x'); ax3d.set_ylabel('m_y'); ax3d.set_zlabel('m_z')
        ax3d.set_title(f'LLG Magnetization Precession, timestep={self.dt} s, steps={self.n_steps}, damping={self.alpha}')
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        # ax3d.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                        # color='lightgray', alpha=0.3)
        ax3d.quiver(0, 0, 0, *self.Hext, color='red', normalize=False,  arrow_length_ratio=0.08)
        ax3d.text(*(self.Hext * 1.1)-np.array([0.03, 0, 0]), r'$\mathbf{H}_{eff}$', color='red', fontsize=20)        
        ax3d.quiver(0, 0, 0, *self.m0, color='blue', normalize=False,  arrow_length_ratio=0.08)
        ax3d.text(*(self.m0 * 0.5)+np.array([0.05, 0, 0]), r'$\mathbf{M}_0$', color='blue', fontsize=20)
        mx, my, mz = self.m0
        m0 = self.m0
        Heff = self.Hext   # or self.Heff if you have it

        v_prec = -np.cross(self.m0, Heff)
        v_damp = np.cross(self.m0, v_prec)
        ax3d.quiver(
            mx, my, mz,
            *v_damp,
            color='purple',
            normalize=True,
            length=0.3
        )
        ax3d.text(
            *(m0 + 0.35 * v_damp / np.linalg.norm(v_damp)),
            r'$ \mathbf{M}\times\frac{d\mathbf{M}}{dt}$',
            color='purple', fontsize=20
        )

        ax3d.quiver(
            mx, my, mz,           # origin = tip of m0
            *v_prec,              # direction
            color='green',
            normalize=True,
            length=0.3, 
        )
        ax3d.text(
            *(m0 + 0.32 * v_prec / np.linalg.norm(v_prec)),
            r'$- \mathbf{M}\times\mathbf{H}_{eff}$',
            color='green', fontsize=20
        )
        ax3d.grid(False)
        plt.tight_layout()
        # 3D vector and trace
        # vector, = ax3d.quiver([], [], [], 'o-', color='blue', lw=3)
        trace, = ax3d.plot([], [], [], color='cyan', alpha=0.6)

        def init():
            # vector.set_data([], [])
            # vector.set_3d_properties([])
            trace.set_data([], [])
            trace.set_3d_properties([])

            return (trace,) 

        def update(frame):
            # Update 3D vector
            mx, my, mz = self.m_steps[frame]
            # vector.set_data([0, mx], [0, my])
            # vector.set_3d_properties([0, mz])
            trace.set_data(self.m_steps[:frame,0], self.m_steps[:frame,1])
            trace.set_3d_properties(self.m_steps[:frame,2])

            return (trace,) 

        ani = FuncAnimation(fig, update,
                        frames=range(0, len(self.m_steps)),
                        init_func=init,
                        interval=1,
                        blit=True)
    
        ani.save("llg_precession.gif", writer='pillow', fps=10,)
        plt.savefig("llg_precession.png", transparent=True)
        plt.close(fig)
    

    def plot_m_vector(self):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
        ax.set_xlabel('m_x'); ax.set_ylabel('m_y'); ax.set_zlabel('m_z')
        ax.set_title('LLG Magnetization Precession')

        # draw unit sphere for context
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                        color='lightgray', alpha=0.3)
        ax.grid(True)

        # draw external field arrow
        ax.quiver(0, 0, 0, *self.Hext, color='red', length=0.8, normalize=True)
        return fig, ax
    
    def llg_dm_dt(self, m, H_eff):
        pre = self.gamma / (1 + self.alpha**2)
        mxh = np.cross(m, H_eff)
        # print('mxh', mxh)
        mxmxh = np.cross(m, mxh)
        return -pre * mxh - (pre * self.alpha * mxmxh)/self.Ms 
    
    def effective_field(self, m, t):
        return self.Hext 
    
    def rk4_step(self, m, i):
        H = self.effective_field(m, i * self.dt)
        k1 = self.llg_dm_dt(m, H)
        k2 = self.llg_dm_dt(m + 0.5*self.dt*k1, H)
        k3 = self.llg_dm_dt(m + 0.5*self.dt*k2, H)
        k4 = self.llg_dm_dt(m + self.dt*k3, H)
        m_new = m + (self.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        return m_new / np.linalg.norm(m_new)
    
    # ---------- Unnecessary functions for this simulation ----------
    def m_components(self, m, theta, phi):
        mx = m * np.sin(theta) * np.cos(phi)
        my = m * np.sin(theta) * np.sin(phi)
        mz = m * np.cos(theta) 
        return mx, my, mz

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

if __name__ == "__main__":
	LLG = LLGsim(Hext=np.array([0, 0, 1]), m0=np.array([0, 0.1, -1]), alpha=0.1, gamma=1.76e11, dt=1e-12, n_steps=3500)

	LLG.sim_m_vector()
	print(LLG.m_steps)
	LLG.animate_m_vector()
	print('animating')
	LLG.plot_m_components()
