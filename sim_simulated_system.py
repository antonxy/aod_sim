import multiprocessing
import numpy as np
import scipy
import scipy.io
import scipy.ndimage
from numpy import exp, pi, sqrt, log2, arccos
from scipy.ndimage import gaussian_filter
import numpy.fft as fft
from numpy import newaxis
import skimage
import skimage.draw
import time
import random

import matplotlib.pyplot as plt

def fft_downsample(img, block_size=2):
    imf = fft.fft2(img)
    assert img.shape[0] == img.shape[1]
    N2 = img.shape[0] // block_size
    f = block_size * 2 - 1
    imf_down = np.zeros((N2, N2), np.complex128)
    imf_down[:N2//2, :N2//2] = imf[:N2//2, :N2//2]
    imf_down[:N2//2, N2//2:] = imf[:N2//2, f * N2//2:]
    imf_down[N2//2:, N2//2:] = imf[f * N2//2:, f * N2//2:]
    imf_down[N2//2:, :N2//2] = imf[f * N2//2:, :N2//2]
    return np.real(fft.ifft2(imf_down))

class SIMSimulatedSystem:
    def __init__(self):
        self.N = 256
        # Optics parameters
        pixelsize = 11  # camera pixel size, um
        magnification = 1.5  # objective magnification
        NA = 0.02  # numerial aperture at sample
        n = 1  # refractive index at sample
        wavelength = 0.471  # wavelength, um

        self._dx = pixelsize / magnification  # Sampling in image plane
        self._res = wavelength / (2 * NA)
        self.illumination_res = self._res # Illumination could have a different PSF as imaging
        _oversampling = self._res / self._dx
        _dk = _oversampling / (self.N / 2)  # Sampling in frequency plane

        self.sample_width = self.N * self._dx

        self.sim_oversample = 4
        self.sim_N = self.N * self.sim_oversample
        self.sim_xy = self.coords(self.sim_N)

        self.simulate_grating = True
        self.grating_repeats = 5
        self.grating_distance_deg = 0.5
        self.grating_angle_deg = 10.0

    def coords(self, N):
        # 1d array along x and y
        sim_x = np.linspace(-self.sample_width/2, self.sample_width/2, N)
        sim_y = sim_x

        # separate 2d arrays of x and y coordinates
        sim_xx, sim_yy = np.meshgrid(sim_x, sim_y)

        # 2d array of 2d vectors [x, y]
        sim_xy = np.zeros((N, N, 2))
        sim_xy[:, :, 0] = sim_xx
        sim_xy[:, :, 1] = sim_yy
        return sim_xy

    def connect(self):
        self.O = self.sample()

    def disconnect(self):
        pass

    def configure_camera(self, exposure_time_sec):
        pass

    def sample(self):
        O = np.zeros((self.sim_N, self.sim_N))
        #O[:, :] = 1.0
        #return O

        for i in range(10000):
            center = (random.random() * self.sim_N for i in range(2))
            rr, cc = skimage.draw.disk(center, 5 / self._dx * self.sim_oversample, shape=O.shape)
            O[rr, cc] += 1.0
        rr, cc = skimage.draw.disk((self.sim_N / 2, self.sim_N / 2), 200 / self._dx * self.sim_oversample, shape=O.shape)
        O[rr, cc] += 1.0
        return O

    def grating(self):
        grating_dots = np.linspace(-self.grating_distance_deg * (self.grating_repeats - 1) / 2, self.grating_distance_deg * (self.grating_repeats - 1) / 2, self.grating_repeats)
        dots_xx, dots_yy = np.meshgrid(grating_dots, grating_dots)
        dots_xy = np.zeros((self.grating_repeats, self.grating_repeats, 2))
        dots_xy[:, :, 0] = dots_xx
        dots_xy[:, :, 1] = dots_yy
        dots_xy = dots_xy.reshape((-1, 2))

        # Apply orientation
        orientation_rad = np.deg2rad(self.grating_angle_deg)
        c, s = np.cos(orientation_rad), np.sin(orientation_rad)
        R = np.array(((c, -s), (s, c)))
        dots_xy = np.dot(dots_xy, R.T)

        return dots_xy

    def illumination(self, pattern_deg):
        if self.simulate_grating:
            pattern_deg = pattern_deg[np.newaxis, :, :] + self.grating()[:, np.newaxis, :]
            pattern_deg = pattern_deg.reshape((-1, 2))

        deg_to_um = 700.0
        center_um = np.array([self.sample_width / 2] * 2)
        pattern_um = (pattern_deg * deg_to_um) + center_um

        pattern_idx = np.round(pattern_um / self._dx * self.sim_oversample).astype(np.int32)
        pattern_idx[:,[0, 1]] = pattern_idx[:,[1, 0]] # swap x, y to row, col

        I = np.zeros((self.sim_N, self.sim_N))
        I.ravel()[np.ravel_multi_index(pattern_idx.T, I.shape, mode='clip')] = 1.0

        return gaussian_filter(I, self.illumination_res/self._dx*self.sim_oversample/2)

    def project_patterns_and_take_images(self, patterns_deg, pattern_rate_Hz):
        O = self.O
        Im = np.zeros((7, self.N, self.N))
        for i in range(7):
            pattern_deg = patterns_deg[i, :, :]
            I_n = self.illumination(pattern_deg)
            Im[i] = fft_downsample(gaussian_filter(I_n * O, self._res/self._dx*self.sim_oversample/2), block_size=self.sim_oversample)
        return Im

    def project_patterns_looping(self, patterns_deg, pattern_rate_Hz, run_event):
        while run_event.is_set():
            print("Project")
            time.sleep(1)
