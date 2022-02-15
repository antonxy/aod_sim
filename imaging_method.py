from PySide2 import QtCore, QtWidgets, QtGui
from widgets import MplCanvas, PlotWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import hex_grid
import lmi_pattern
from hexSimProcessor import HexSimProcessor
import scipy
import tifffile
import os
import json

class ImagingMethod:
    def __init__(self, method_name):
        self.method_name = method_name
        self.parameters_widget = None
        self.reconstruction_parameters_widget = None
        self.patterns_widget = None
        self.debug_tabs = []  # list of tuples ("name", widget)
        self.frames = None
        self.reconstruction = None
        self.params = None

    def parse_parameters(self):
        pass

    def update_patterns(self, global_params):
        pass

    def take_images(self, system):
        pass

    def calibrate(self):
        pass

    def reconstruct(self):
        pass

    def save_images(self, folder):
        pass

    def load_images(self, folder):
        pass

    def save_images(self, folder):
        tifffile.imwrite(os.path.join(folder, f"{self.method_name}.tiff"), self.frames)
        if self.reconstruction is not None:
            tifffile.imwrite(os.path.join(folder, f"{self.method_name}_reconstruction.tiff"), self.reconstruction)
        with open(os.path.join(folder, f"{self.method_name}_metadata.json"), 'w') as f:
            json.dump(self.params, f)

    def load_images(self, folder):
        self.frames = tifffile.imread(os.path.join(folder, f"{self.method_name}.tiff"))
        self.reconstruction = None
        self.plot_images()


class SIMImaging(ImagingMethod):
    def __init__(self):
        super().__init__("sim")
        sim_group = QtWidgets.QGroupBox("SIM")
        layout = QtWidgets.QFormLayout()

        self.sim_enabled_chb = QtWidgets.QCheckBox("Do SIM")
        self.sim_enabled_chb.setChecked(True)
        layout.addRow("Do SIM", self.sim_enabled_chb)

        self.desired_distance_txt = QtWidgets.QLineEdit("0.0345")
        layout.addRow("Desired dot distance [deg]", self.desired_distance_txt)

        self.steps_x_lbl = QtWidgets.QLabel()
        layout.addRow("Steps X", self.steps_x_lbl)

        self.steps_y_lbl = QtWidgets.QLabel()
        layout.addRow("Steps Y", self.steps_y_lbl)

        self.dot_distance_x_lbl = QtWidgets.QLabel()
        layout.addRow("Dot distance x", self.dot_distance_x_lbl)

        self.aspect_lbl = QtWidgets.QLabel()
        layout.addRow("Aspect ratio", self.aspect_lbl)

        self.pattern_hz_txt = QtWidgets.QLineEdit("40000")
        layout.addRow("Projection rate [Hz]", self.pattern_hz_txt)

        self.exposure_lbl = QtWidgets.QLabel()
        layout.addRow("Exposure time", self.exposure_lbl)
        sim_group.setLayout(layout)

        self.parameters_widget = sim_group

        reconstruction_group = QtWidgets.QGroupBox("Reconstruction")
        layout = QtWidgets.QFormLayout()

        self.reconstruction_size_txt = QtWidgets.QLineEdit("256")
        layout.addRow("Reconstruction size N", self.reconstruction_size_txt)

        self.reconstruction_offset_x = QtWidgets.QLineEdit("0")
        layout.addRow("Reconstruction offset x", self.reconstruction_offset_x)

        self.reconstruction_offset_y = QtWidgets.QLineEdit("0")
        layout.addRow("Reconstruction offset y", self.reconstruction_offset_y)

        self.reconstruction_eta_txt = QtWidgets.QLineEdit("0.5")
        layout.addRow("Reconstruction eta", self.reconstruction_eta_txt)

        self.use_filter_chb = QtWidgets.QCheckBox("Use frequency space filtering")
        layout.addRow("Filter", self.use_filter_chb)

        reconstruction_group.setLayout(layout)
        self.reconstruction_parameters_widget = reconstruction_group

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout()
        self.pattern_plot = MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(self.pattern_plot, w)
        lay.addWidget(toolbar)
        lay.addWidget(self.pattern_plot)
        w.setLayout(lay)
        self.patterns_widget = w

        self.image_plot = PlotWidget()
        self.fft_plot = PlotWidget()
        self.carrier_plot = PlotWidget()
        self.bands_plot = PlotWidget()
        self.psf_plot = PlotWidget()
        self.recon_plot = PlotWidget(None, num_clim = 2)

        self.debug_tabs = [
            ('Recorded Images', self.image_plot),
            ('FFT', self.fft_plot),
            ('Carrier', self.carrier_plot),
            ('Bands', self.bands_plot),
            ('PSF shaping', self.psf_plot),
            ('Reconstructed Image', self.recon_plot),
        ]

        self.p = HexSimProcessor()
        self.p.debug = False

        self.frames = None
        self.reconstruction = None

    def parse_parameters(self):
        return {
            "desired_distance": float(self.desired_distance_txt.text()),
            "pattern_rate_Hz": float(self.pattern_hz_txt.text()),
        }

    def update_patterns(self, global_params):
        if not self.sim_enabled_chb.isChecked():
            return
        params = {**self.parse_parameters(), **global_params}

        desired_distance = params['desired_distance']
        grating_distance_x = params['grating_distance_x']
        steps_x = round(grating_distance_x / desired_distance)
        distance_x = grating_distance_x / steps_x

        grating_distance_y = params['grating_distance_y']
        desired_distance_y = np.sin(np.deg2rad(60)) * distance_x * 2
        steps_y = round(grating_distance_y / desired_distance_y)
        distance_y = grating_distance_y / steps_y

        aspect = grating_distance_y / (desired_distance_y * steps_y)
        self.aspect_lbl.setText(str(aspect))

        self.steps_x_lbl.setText(str(steps_x))
        self.steps_y_lbl.setText(str(steps_y))
        self.dot_distance_x_lbl.setText(str(distance_x))

        orientation_deg = params['orientation_deg']
        self.pattern_rate_Hz = params['pattern_rate_Hz']

        pattern_deg = hex_grid.projection_hex_pattern_deg(distance_x, steps_x, steps_y, orientation_rad = np.deg2rad(orientation_deg), aspect_ratio=aspect)

        self.exposure_time_sec = pattern_deg.shape[1] / self.pattern_rate_Hz
        self.exposure_lbl.setText(f"{self.exposure_time_sec * 1e3:.1f} ms * 7 = {self.exposure_time_sec * 1e3 * 7:.1f} ms")

        self.pattern_plot.fig.clear()
        ax1, ax2 = self.pattern_plot.fig.subplots(1, 2, sharex=True, sharey=True)
        for i in range(7):
            ax1.scatter(pattern_deg[i, :, 0], pattern_deg[i, :, 1])
        ax2.scatter(pattern_deg[0, :, 0], pattern_deg[0, :, 1])
        ax1.set_aspect(1)
        ax2.set_aspect(1)
        self.pattern_plot.draw()

        self.pattern_deg = pattern_deg
        self.params = params
        self.update_psf_plots()

    def update_psf_plots(self):
        desired_distance = self.params['desired_distance']
        aod_deg_to_um_in_sample_plane = self.params['aod_deg_to_um_in_sample_plane']
        pixelsize = self.params['pixelsize']
        magnification = self.params['magnification']
        NA = self.params['NA']
        wavelength = self.params['wavelength']
        mtf_data = self.params['mtf_data']

        # TODO could do all this in pattern calculation also, would make more sense
        #Expected PSF
        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        def plot_psf(plt, k):
            N = 9
            dx = pixelsize / magnification
            res = wavelength / (2 * NA)
            x = np.linspace(-dx*N/2, dx*N/2, N * 10)  # [um]
            sigma = res / 2.355 # FWHM to sigma

            # TODO res is not FWHM, or is it?. And PSF is not really gaussian but a sinc
            psf_orig = gaussian(x, 0, sigma)
            plt.plot(x, psf_orig, label="emission psf")

            mod = (1 + np.cos(k * x)) / 2
            plt.plot(x, mod, '--', label="modulation")

            plt.plot(x, psf_orig * mod, label="SIM psf")

            plt.vlines(np.arange(-dx*N/2, dx*N/2, dx), 0, 0.1, 'r', label="pixel in sample plane")

            plt.set_xlabel("x [um]")
            plt.set_title("PSF Shaping")
            plt.legend(loc="upper right")

        def plot_otf(plt, k):
            N = 100
            dx = pixelsize / magnification
            res = wavelength / (2 * NA)
            oversampling = res / dx
            dk = oversampling / (N / 2)  # Sampling in frequency plane. This is unitless, not sure what is means exactly
            k_to_cycles_per_um = NA / wavelength
            kx = np.arange(-dk * N / 2, dk * N / 2, dk)

            plt.plot(kx * k_to_cycles_per_um, self.p._tf(abs(kx)) * 2, label="OTF");
            nyq = 0.5 / dx
            plt.vlines([-nyq, nyq], 0, 0.2, 'r', label="nyquist frequency")
            plt.vlines([-k / (2 * np.pi), k / (2 * np.pi)], 0, 0.2, 'g', label="illumination frequency")
            if mtf_data != "":
                mtf = np.fromstring("\n".join(mtf_data.split("\n")[1:]), sep='\t').reshape(-1, 2)
                plt.plot(mtf[:, 0] / 1000, mtf[:, 1], label="measured MTF")
            plt.set_ylim(0, None)
            plt.set_xlabel("Frequency [cycles / um]")
            plt.legend(loc="upper right")

        illumination_freq = (1 / (desired_distance * aod_deg_to_um_in_sample_plane)) * 2 * np.pi

        self.psf_plot.plot.fig.clear()
        axs = self.psf_plot.plot.fig.subplots(1, 2)
        plot_psf(axs[0], k = illumination_freq)
        plot_otf(axs[1], k = illumination_freq)
        self.psf_plot.plot.draw()

    def take_images(self, system):
        if not self.sim_enabled_chb.isChecked():
            return
        self.frames = system.project_patterns_and_take_images(self.pattern_deg, self.pattern_rate_Hz, self.params['pattern_delay_sec'])
        self.reconstruction = None
        self.plot_images()

    def plot_images(self):
        self.image_plot.plot.fig.clear()
        axs = self.image_plot.plot.fig.subplots(3, 3, sharex=True, sharey=True)
        ims = []
        for i in range(3):
            ims.append(axs[0, i].imshow(self.frames[i]))
        for i in range(3):
            ims.append(axs[1, i].imshow(self.frames[i + 3]))
        ims.append(axs[2, i].imshow(self.frames[6]))
        self.image_plot.connect_clim(ims)
        self.image_plot.plot.draw()

        self.fft_plot.plot.clear()
        im = self.fft_plot.plot.axes.imshow(np.abs(np.fft.fftshift(np.fft.fft2(self.frames[0]))))
        self.fft_plot.connect_clim(im)
        self.fft_plot.plot.draw()

    def calibrate(self):
        if not self.sim_enabled_chb.isChecked():
            return
        N = int(self.reconstruction_size_txt.text())
        offset_x = int(self.reconstruction_offset_x.text())
        offset_y = int(self.reconstruction_offset_y.text())
        eta = float(self.reconstruction_eta_txt.text())
        pixelsize = self.params['pixelsize']
        magnification = self.params['magnification']
        NA = self.params['NA']
        wavelength = self.params['wavelength']
        mtf_data = self.params['mtf_data']


        assert self.frames.shape[0] == 7
        frames = self.frames[:, offset_y:offset_y + N, offset_x:offset_x + N]

        self.p.N = N
        self.p.pixelsize = pixelsize
        self.p.magnification = magnification
        self.p.NA = NA
        self.p.wavelength = wavelength
        self.p.eta = eta
        self.p.calibrate(frames)

        self.carrier_plot.plot.fig.clear()
        axs = self.carrier_plot.plot.fig.subplots(2, 3)
        ims = []
        for i in range(3):
            ims.append(axs[0, i].imshow(self.p.carrier_debug_img[i]))
        for i in range(3):
            ims.append(axs[1, i].imshow(self.p.carrier_debug_zoom_img[i]))
        self.carrier_plot.connect_clim(ims)
        self.carrier_plot.plot.draw()

        self.bands_plot.plot.fig.clear()
        axs = self.bands_plot.plot.fig.subplots(2, 4)
        ims = []
        for i in range(4):
            ims.append(axs[0, i].imshow(self.p.bands_debug_img[i].real))
        for i in range(4):
            ims.append(axs[1, i].imshow(self.p.bands_debug_img[i].imag))
        self.bands_plot.connect_clim(ims)
        self.bands_plot.plot.draw()


    def reconstruct(self):
        if not self.sim_enabled_chb.isChecked():
            return None, None
        # TODO this breaks if params changed inbetween
        N = int(self.reconstruction_size_txt.text())
        offset_x = int(self.reconstruction_offset_x.text())
        offset_y = int(self.reconstruction_offset_y.text())

        assert self.frames.shape[0] == 7
        frames = self.frames[:, offset_y:offset_y + N, offset_x:offset_x + N]

        self.p.use_filter = self.use_filter_chb.isChecked()
        reconstruct = self.p.reconstruct_fftw(frames)
        sumall = scipy.ndimage.zoom(np.sum(frames, axis=0), (2, 2), order=1)

        self.recon_plot.plot.fig.clear()
        ax1, ax2 = self.recon_plot.plot.fig.subplots(1, 2, sharex=True, sharey=True)
        im1 = ax1.imshow(reconstruct)
        im2 = ax2.imshow(sumall)
        self.recon_plot.connect_clim(im1, 0)
        self.recon_plot.connect_clim(im2, 1)
        self.recon_plot.plot.draw()

        self.reconstruction = reconstruct

        return reconstruct, sumall

    def save_images(self, folder):
        if self.sim_enabled_chb.isChecked():
            super().save_images(folder)

    def load_images(self, folder):
        if self.sim_enabled_chb.isChecked():
            super().load_images(folder)


class LMIImaging(ImagingMethod):
    def __init__(self):
        super().__init__("lmi")
        lmi_group = QtWidgets.QGroupBox("LMI")
        layout = QtWidgets.QFormLayout()

        self.lmi_enabled_chb = QtWidgets.QCheckBox("Do LMI")
        self.lmi_enabled_chb.setChecked(True)
        layout.addRow("Do LMI", self.lmi_enabled_chb)

        self.steps_x_txt = QtWidgets.QLineEdit("5")
        layout.addRow("Steps X", self.steps_x_txt)

        self.steps_y_txt = QtWidgets.QLineEdit("5")
        layout.addRow("Steps Y", self.steps_y_txt)

        self.multiscan_x_txt = QtWidgets.QLineEdit("10")
        layout.addRow("Multiscan X", self.multiscan_x_txt)

        self.multiscan_y_txt = QtWidgets.QLineEdit("10")
        layout.addRow("Multiscan Y", self.multiscan_y_txt)

        self.dot_distance_x_lbl = QtWidgets.QLabel()
        layout.addRow("Dot distance X", self.dot_distance_x_lbl)

        self.dot_distance_y_lbl = QtWidgets.QLabel()
        layout.addRow("Dot distance Y", self.dot_distance_y_lbl)

        self.multiscan_distance_x_lbl = QtWidgets.QLabel()
        layout.addRow("Multiscan distance X", self.multiscan_distance_x_lbl)

        self.multiscan_distance_y_lbl = QtWidgets.QLabel()
        layout.addRow("Multiscan distance Y", self.multiscan_distance_y_lbl)

        self.pattern_hz_txt = QtWidgets.QLineEdit("40000")
        layout.addRow("Projection rate [Hz]", self.pattern_hz_txt)

        self.exposure_lbl = QtWidgets.QLabel()
        layout.addRow("Exposure time", self.exposure_lbl)

        lmi_group.setLayout(layout)

        self.parameters_widget = lmi_group

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout()
        self.pattern_plot = MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(self.pattern_plot, w)
        lay.addWidget(toolbar)
        lay.addWidget(self.pattern_plot)
        w.setLayout(lay)
        self.patterns_widget = w

        self.image_plot = PlotWidget()
        self.recon_plot = PlotWidget(None, num_clim = 2)
        self.debug_tabs = [
            ('LMI Recorded Images', self.image_plot),
            ('LMI Reconstructed Image', self.recon_plot),
        ]

    def parse_parameters(self):
        return {
            "steps_x": int(self.steps_x_txt.text()),
            "steps_y": int(self.steps_y_txt.text()),
            "multiscan_x": int(self.multiscan_x_txt.text()),
            "multiscan_y": int(self.multiscan_y_txt.text()),
            "pattern_rate_Hz": float(self.pattern_hz_txt.text()),
        }

    def update_patterns(self, global_params):
        if not self.lmi_enabled_chb.isChecked():
            return

        params = {**self.parse_parameters(), **global_params}

        grating_distance_x = params['grating_distance_x']
        grating_distance_y = params['grating_distance_y']
        orientation_deg = params['orientation_deg']

        steps_x = params['steps_x']
        steps_y = params['steps_y']

        multiscan_x = params['multiscan_x']
        multiscan_y = params['multiscan_y']

        self.pattern_rate_Hz = params['pattern_rate_Hz']

        pattern_deg = lmi_pattern.lmi_pattern_deg(steps_x, steps_y, multiscan_x, multiscan_y, grating_distance_x, grating_distance_y, orientation_rad = np.deg2rad(orientation_deg))

        self.exposure_time_sec = pattern_deg.shape[1] / self.pattern_rate_Hz
        num_patterns = pattern_deg.shape[0]
        self.exposure_lbl.setText(f"{self.exposure_time_sec * 1e3:.1f} ms * {num_patterns} = {self.exposure_time_sec * 1e3 * num_patterns:.1f} ms")
        self.dot_distance_x_lbl.setText(str(grating_distance_x / multiscan_x / steps_x))
        self.dot_distance_y_lbl.setText(str(grating_distance_y / multiscan_y / steps_y))
        self.multiscan_distance_x_lbl.setText(str(grating_distance_x / multiscan_x))
        self.multiscan_distance_y_lbl.setText(str(grating_distance_y / multiscan_y))

        self.pattern_plot.fig.clear()
        ax1, ax2 = self.pattern_plot.fig.subplots(1, 2, sharex=True, sharey=True)
        for i in range(num_patterns):
            ax1.scatter(pattern_deg[i, :, 0], pattern_deg[i, :, 1])
        ax2.scatter(pattern_deg[0, :, 0], pattern_deg[0, :, 1])
        ax1.set_aspect(1)
        ax2.set_aspect(1)
        self.pattern_plot.draw()

        self.pattern_deg = pattern_deg
        self.params = params


    def take_images(self, system):
        if not self.lmi_enabled_chb.isChecked():
            return
        self.frames = system.project_patterns_and_take_images(self.pattern_deg, self.pattern_rate_Hz, self.params['pattern_delay_sec'])
        self.reconstruction = None
        self.plot_images()

    def plot_images(self):
        self.image_plot.plot.fig.clear()
        axs = self.image_plot.plot.fig.subplots(3, 3, sharex=True, sharey=True)
        ims = []
        for i in range(9):
            ims.append(axs[i // 3, i % 3].imshow(self.frames[i]))
        self.image_plot.connect_clim(ims)
        self.image_plot.plot.draw()

    def reconstruct(self):
        if not self.lmi_enabled_chb.isChecked():
            return None, None
        reconstruct = np.max(self.frames, axis=0)
        #reconstruct = np.max(self.frames, axis=0)
        sumall = np.sum(self.frames, axis=0)

        self.recon_plot.plot.fig.clear()
        ax1, ax2 = self.recon_plot.plot.fig.subplots(1, 2, sharex=True, sharey=True)
        im1 = ax1.imshow(reconstruct)
        im2 = ax2.imshow(sumall)
        self.recon_plot.connect_clim(im1, 0)
        self.recon_plot.connect_clim(im2, 1)
        self.recon_plot.plot.draw()

        self.reconstruction = reconstruct

        return reconstruct, sumall

    def save_images(self, folder):
        if self.lmi_enabled_chb.isChecked():
            super().save_images(folder)

    def load_images(self, folder):
        if self.lmi_enabled_chb.isChecked():
            super().load_images(folder)
