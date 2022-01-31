import nidaqmx
import numpy as np

def angle_deg_to_frequency_MHz(angle_deg):
    angleRange = 2.29183  # 40 mrad  Maximum angle possible by AOD
    F_central = 82
    bandWidth = 50  # MHz, total bandwidth supported by the AOD

    freq_MHz = F_central + ((angle_deg / (angleRange / 2)) * (bandWidth / 2))

    bandWidthCheck = 25  # Could be increased to 50 but set to 25 for now just to be safe
    F_max = F_central + bandWidthCheck/2  # Maximum scanning frequency. The central frequency can be optimized so that the diffraction efficiency is similar within the scanning range
    F_min = F_central - bandWidthCheck/2  # Minimum scanning frequency
    assert((freq_MHz <= F_max).all())
    assert((freq_MHz >= F_min).all())
    return freq_MHz

def frequency_MHz_to_bin(x, y):
    x_int = np.round(x / 500 * (2 ** 15 - 1)).astype(np.uint32)
    y_int = np.round(y / 500 * (2 ** 15 - 1)).astype(np.uint32)

    assert((x_int < 2**15).all())
    assert((y_int < 2**15).all())

    return (((1 << 15 | x_int) << 16) | (1 << 15 | y_int)).astype(np.uint32)

def main():
    l = np.linspace(0, 2*np.pi, 500)
    scan_angle_x = angle_deg_to_frequency_MHz(np.sin(l) * 0.5)
    scan_angle_y = angle_deg_to_frequency_MHz(np.cos(l) * 0.5)
    scan = frequency_MHz_to_bin(scan_angle_x, scan_angle_y)
    num_positions = scan.shape[0]

    sys = nidaqmx.system.System.local()
    dev = sys.devices['Dev1']
    while True:
        try:
            print("-")
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan('Dev1/port0:Dev1/port3', line_grouping=nidaqmx.constants.LineGrouping.CHAN_FOR_ALL_LINES)
                task.timing.cfg_samp_clk_timing(rate=20000.0, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=num_positions)
                task.export_signals.export_signal(nidaqmx.constants.Signal.START_TRIGGER, '/Dev1/PFI4')
                task.write(scan, auto_start=True)
                task.wait_until_done()
        except KeyboardInterrupt:
            break

    dev.reset_device()


if __name__ == "__main__":
    main()
