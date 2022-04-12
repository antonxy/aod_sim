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

    def load_parameters(self, params):
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

    def save_images(self, folder, save_raw = True, save_reconstruction = True, save_metadata = True, prefix = ""):
        if save_raw:
            tifffile.imwrite(os.path.join(folder, f"{prefix}{self.method_name}.tiff"), self.frames)
        if self.reconstruction is not None and save_reconstruction:
            tifffile.imwrite(os.path.join(folder, f"{prefix}{self.method_name}_reconstruction.tiff"), self.reconstruction)
        if save_metadata:
            with open(os.path.join(folder, f"{prefix}{self.method_name}_metadata.json"), 'w') as f:
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

        self.reconstruction_w_txt = QtWidgets.QLineEdit("0.3")
        layout.addRow("Wiener parameter", self.reconstruction_w_txt)

        self.reconstruction_alpha_txt = QtWidgets.QLineEdit("0.3")
        layout.addRow("Zero order attenuation width", self.reconstruction_alpha_txt)

        self.reconstruction_beta_txt = QtWidgets.QLineEdit("0.999")
        layout.addRow("Zero order attenuation", self.reconstruction_beta_txt)

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

            "reconstruction_size": int(self.reconstruction_size_txt.text()),
            "reconstruction_offset_x": int(self.reconstruction_offset_x.text()),
            "reconstruction_offset_y": int(self.reconstruction_offset_y.text()),
            "reconstruction_eta": float(self.reconstruction_eta_txt.text()),
            "reconstruction_w": float(self.reconstruction_w_txt.text()),
            "reconstruction_alpha": float(self.reconstruction_alpha_txt.text()),
            "reconstruction_beta": float(self.reconstruction_beta_txt.text()),
            "reconstruction_use_filter": self.use_filter_chb.isChecked(),
        }

    def load_parameters(self, params):
        self.desired_distance_txt.setText(str(params.get("desired_distance", "0.013")))
        self.pattern_hz_txt.setText(str(params.get("pattern_rate_Hz", "10000")))

        self.reconstruction_size_txt.setText(str(params.get("reconstruction_size", "1000")))
        self.reconstruction_offset_x.setText(str(params.get("reconstruction_offset_x", "0")))
        self.reconstruction_offset_y.setText(str(params.get("reconstruction_offset_y", "0")))
        self.reconstruction_eta_txt.setText(str(params.get("reconstruction_eta", "0.5")))
        self.reconstruction_w_txt.setText(str(params.get("reconstruction_w", "0.3")))
        self.reconstruction_alpha_txt.setText(str(params.get("reconstruction_alpha", "0.3")))
        self.reconstruction_beta_txt.setText(str(params.get("reconstruction_beta", "0.999")))
        self.use_filter_chb.setChecked(params.get("reconstruction_use_filter", False))

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
        ex_NA = self.params['ex_NA']
        ex_wavelength = self.params['ex_wavelength']
        em_NA = self.params['em_NA']
        em_wavelength = self.params['em_wavelength']
        mtf_data = self.params['mtf_data']

        # TODO could do all this in pattern calculation also, would make more sense
        #Expected PSF
        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        def plot_psf_modulation(plt, wavelength, NA, k):
            N = 9
            dx = pixelsize / magnification
            res = wavelength / (2 * NA)
            x = np.linspace(-dx*N/2, dx*N/2, N * 10)  # [um]
            sigma = res / 2.355 # FWHM to sigma

            # TODO res is not FWHM, or is it?. And PSF is not really gaussian but a sinc
            psf_orig = gaussian(x, 0, sigma)
            plt.plot(x, psf_orig, label="Emission PSF")

            mod = (1 + np.cos(k * x)) / 2
            plt.plot(x, mod, '--', label="Modulation Function")

            plt.plot(x, psf_orig * mod, label="SIM PSF")

            plt.vlines(np.arange(-dx*N/2, dx*N/2, dx), 0, 0.1, 'r', label="Pixel in Sample Plane")

            plt.set_xlabel("x [um]")
            plt.set_title("PSF Shaping")
            plt.legend(loc="upper right")

        def plot_projected_pattern(plt, wavelength, NA, dot_distance_um):
            N = 9
            dx = pixelsize / magnification
            res = wavelength / (2 * NA)
            x = np.linspace(-dx*N/2, dx*N/2, N * 10)  # [um]
            sigma = res / 2.355 # FWHM to sigma

            # TODO res is not FWHM, or is it?. And PSF is not really gaussian but a sinc
            psf_orig = gaussian(x, 0, sigma)
            plt.plot(x, psf_orig, label="Excitation PSF")

            dot_positions = np.arange(-dx*N, dx*N, dot_distance_um)
            dot_positions -= np.mean(dot_positions) # Center at zero
            projected_pattern = np.sum(np.stack([gaussian(x, mean, sigma) for mean in dot_positions]), axis=0)
            plt.plot(x, projected_pattern, label="Projected Pattern")

            plt.set_xlabel("x [um]")
            plt.set_title("Projected Pattern")
            plt.legend(loc="upper right")

        def plot_otf(plt, wavelength, NA, k, mtf_data = "", name = ""):
            N = 100
            dx = pixelsize / magnification
            res = wavelength / (2 * NA)
            oversampling = res / dx
            dk = oversampling / (N / 2)  # Sampling in frequency plane. This is unitless, not sure what is means exactly
            k_to_cycles_per_um = NA / wavelength
            kx = np.arange(-dk * N / 2, dk * N / 2, dk)

            plt.plot(kx * k_to_cycles_per_um, self.p._tf(abs(kx)) * 2, label=f"{name} MTF");
            nyq = 0.5 / dx
            plt.vlines([-nyq, nyq], 0, 0.2, 'r', label="Nyquist Frequency")
            plt.vlines([-k / (2 * np.pi), k / (2 * np.pi)], 0, 0.2, 'g', label="Illumination Frequency")
            if mtf_data != "":
                mtf = np.fromstring("\n".join(mtf_data.split("\n")[1:]), sep='\t').reshape(-1, 2)
                plt.plot(mtf[:, 0] / 1000, mtf[:, 1], label="Measured MTF")
            plt.set_ylim(0, None)
            plt.set_xlabel("Frequency [cycles / um]")
            plt.legend(loc="upper right")
            plt.set_title(f"{name} MTF")

        dot_distance_um = desired_distance * aod_deg_to_um_in_sample_plane
        illumination_freq = (1 / dot_distance_um) * 2 * np.pi

        self.psf_plot.plot.fig.clear()
        axs = self.psf_plot.plot.fig.subplots(2, 2)
        plot_projected_pattern(axs[0, 0], wavelength = ex_wavelength, NA = ex_NA, dot_distance_um = dot_distance_um)
        plot_otf(axs[0, 1], wavelength = ex_wavelength, NA = ex_NA, k = illumination_freq, name="Excitation")

        plot_psf_modulation(axs[1, 0], wavelength = em_wavelength, NA = em_NA, k = illumination_freq)
        plot_otf(axs[1, 1], wavelength = em_wavelength, NA = em_NA, k = illumination_freq, mtf_data = mtf_data, name="Emission")
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
        alpha = float(self.reconstruction_alpha_txt.text())
        beta = float(self.reconstruction_beta_txt.text())
        w = float(self.reconstruction_w_txt.text())
        eta = float(self.reconstruction_eta_txt.text())
        pixelsize = self.params['pixelsize']
        magnification = self.params['magnification']
        em_NA = self.params['em_NA']
        em_wavelength = self.params['em_wavelength']
        mtf_data = self.params['mtf_data']


        assert self.frames.shape[0] == 7
        frames = self.frames[:, offset_y:offset_y + N, offset_x:offset_x + N]

        self.p.N = N
        self.p.pixelsize = pixelsize
        self.p.magnification = magnification
        self.p.NA = em_NA
        self.p.wavelength = em_wavelength
        self.p.alpha = alpha
        self.p.beta = beta
        self.p.w = w
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

    def save_images(self, *args, **kwargs):
        if self.sim_enabled_chb.isChecked():
            super().save_images(*args, **kwargs)

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

    def load_parameters(self, params):
        self.steps_x_txt.setText(str(params.get("steps_x", "10")))
        self.steps_y_txt.setText(str(params.get("steps_y", "10")))
        self.multiscan_x_txt.setText(str(params.get("multiscan_x", "10")))
        self.multiscan_y_txt.setText(str(params.get("multiscan_y", "10")))
        self.pattern_hz_txt.setText(str(params.get("pattern_rate_Hz", "10000")))

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

    def save_images(self, *args, **kwargs):
        if self.lmi_enabled_chb.isChecked():
            super().save_images(*args, **kwargs)

    def load_images(self, folder):
        if self.lmi_enabled_chb.isChecked():
            super().load_images(folder)


class LineLMIImaging(ImagingMethod):
    def __init__(self):
        super().__init__("lmi")
        lmi_group = QtWidgets.QGroupBox("Line LMI")
        layout = QtWidgets.QFormLayout()

        self.lmi_enabled_chb = QtWidgets.QCheckBox("Do Line LMI")
        self.lmi_enabled_chb.setChecked(True)
        layout.addRow("Do LMI", self.lmi_enabled_chb)

        self.two_grating_chb = QtWidgets.QCheckBox("Two gratings")
        layout.addRow("Two gratings", self.two_grating_chb)

        self.steps_txt = QtWidgets.QLineEdit("5")
        layout.addRow("Steps", self.steps_txt)

        self.multiscan_txt = QtWidgets.QLineEdit("10")
        layout.addRow("Multiscan", self.multiscan_txt)
        
        self.pattern_repeat_txt = QtWidgets.QLineEdit("1")
        layout.addRow("Pattern repeat", self.pattern_repeat_txt)

        self.dot_distance_lbl = QtWidgets.QLabel()
        layout.addRow("Dot distance", self.dot_distance_lbl)

        self.multiscan_distance_lbl = QtWidgets.QLabel()
        layout.addRow("Multiscan distance", self.multiscan_distance_lbl)

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
            "two_gratings": bool(self.two_grating_chb.isChecked()),
            "steps": int(self.steps_txt.text()),
            "multiscan": int(self.multiscan_txt.text()),
            "pattern_repeat": int(self.pattern_repeat_txt.text()),
            "pattern_rate_Hz": float(self.pattern_hz_txt.text()),
        }

    def load_parameters(self, params):
        self.two_grating_chb.setChecked(bool(params.get("two_gratings", False)))
        self.steps_txt.setText(str(params.get("steps", "10")))
        self.multiscan_txt.setText(str(params.get("multiscan", "10")))
        self.pattern_repeat_txt.setText(str(params.get("pattern_repeat", "1")))
        self.pattern_hz_txt.setText(str(params.get("pattern_rate_Hz", "10000")))

    def update_patterns(self, global_params):
        if not self.lmi_enabled_chb.isChecked():
            return

        params = {**self.parse_parameters(), **global_params}

        grating_dot_distances = params['grating_dot_distances']
        orientation_deg = params['orientation_deg']
        distance_between_gratings = params["distance_between_gratings"]
        two_gratings = params['two_gratings']

        steps = params['steps']
        multiscan = params['multiscan']
        pattern_repeat = params['pattern_repeat']

        self.pattern_rate_Hz = params['pattern_rate_Hz']

        if two_gratings:
            pattern_deg = lmi_pattern.line_lmi_pattern_two_grating(steps, multiscan, grating_dot_distances, distance_between_gratings, orientation_deg=orientation_deg)
        else:
            pattern_deg = lmi_pattern.line_lmi_pattern_deg(steps, multiscan, grating_dot_distances, distance_between_gratings, orientation_rad=np.deg2rad(orientation_deg))
        pattern_deg = np.tile(pattern_deg, [1, pattern_repeat, 1])

        self.exposure_time_sec = pattern_deg.shape[1] / self.pattern_rate_Hz
        num_patterns = pattern_deg.shape[0]
        self.exposure_lbl.setText(f"{self.exposure_time_sec * 1e3:.1f} ms * {num_patterns} = {self.exposure_time_sec * 1e3 * num_patterns:.1f} ms")
        self.dot_distance_lbl.setText(str([grating_dot_distance / multiscan / steps for grating_dot_distance in grating_dot_distances]))
        self.multiscan_distance_lbl.setText(str([grating_dot_distance / multiscan for grating_dot_distance in grating_dot_distances]))

        self.pattern_plot.fig.clear()
        ax1, ax2 = self.pattern_plot.fig.subplots(1, 2, sharex=True, sharey=True)
        for i in range(num_patterns):
            ax1.scatter(pattern_deg[i, :, 0], pattern_deg[i, :, 1])
        ax2.scatter(pattern_deg[0, :, 0], pattern_deg[0, :, 1])
        ax1.set_aspect(1)
        ax2.set_aspect(1)
        self.pattern_plot.draw()

        self.pattern_deg = pattern_deg
        self.num_gratings = 2 if two_gratings else 3
        self.params = params


    def take_images(self, system):
        if not self.lmi_enabled_chb.isChecked():
            return
        self.frames = system.project_patterns_and_take_images(self.pattern_deg, self.pattern_rate_Hz, self.params['pattern_delay_sec'])
        self.reconstruction = None
        self.plot_images()

    def plot_images(self):
        self.image_plot.plot.fig.clear()
        axs = self.image_plot.plot.fig.subplots(self.num_gratings, 3, sharex=True, sharey=True)
        ims = []
        frames_per_grating = int(len(self.frames) / 3)
        for i in range(self.num_gratings):
            for j in range(3):
                ims.append(axs[i, j].imshow(self.frames[i * frames_per_grating + j]))
        self.image_plot.connect_clim(ims)
        self.image_plot.plot.draw()

    def reconstruct(self):
        if not self.lmi_enabled_chb.isChecked():
            return None, None
        pass

    def save_images(self, *args, **kwargs):
        if self.lmi_enabled_chb.isChecked():
            super().save_images(*args, **kwargs)

    def load_images(self, folder):
        if self.lmi_enabled_chb.isChecked():
            super().load_images(folder)
