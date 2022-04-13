import numpy as np
import threading
import time

def angle_deg_to_frequency_MHz(angle_deg):
    angleRange = 2.29183  # 40 mrad  Maximum angle possible by AOD
    F_central = 82
    bandWidth = 50  # MHz, total bandwidth supported by the AOD

    freq_MHz = F_central + ((angle_deg / (angleRange / 2)) * (bandWidth / 2))

    bandWidthCheck = 50  # Could be increased to 50 but set to 25 for now just to be safe
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


def save_pattern_as_csv(pattern, filename):
    pattern = pattern.reshape(-1, 2)
    scan_angle_x = angle_deg_to_frequency_MHz(pattern[:, 0])
    scan_angle_y = angle_deg_to_frequency_MHz(pattern[:, 1])
    scan = frequency_MHz_to_bin(scan_angle_x, scan_angle_y)

    with open(filename, 'w') as f:
        for i in range(pattern.shape[0]):
            binary = np.binary_repr(scan[i], width=32)
            assert(len(binary) == 32)
            print(';'.join(binary), file=f)


def project_patterns(patterns_degree, rate, reset_when_done = True, loop = False, loop_event = None, export_clock = False):
    import nidaqmx
    # patterns_degree dimensions: [pattern, sample in pattern, axis (x,y)]
    if len(patterns_degree.shape) == 2:
        patterns_degree = patterns_degree[np.newaxis, :, :]
    assert(len(patterns_degree.shape) == 3)
    assert(patterns_degree.shape[2] == 2)

    num_patterns = patterns_degree.shape[0]
    num_samples = patterns_degree.shape[1]

    scan_angle_x = angle_deg_to_frequency_MHz(patterns_degree[:, :, 0])
    scan_angle_y = angle_deg_to_frequency_MHz(patterns_degree[:, :, 1])
    scan = frequency_MHz_to_bin(scan_angle_x, scan_angle_y)

    print(f"Projecting {num_patterns} patterns with {num_samples} samples each")

    sys = nidaqmx.system.System.local()
    dev = sys.devices['Dev1']

    # if no loop event set just loop forever
    if loop_event is None:
        loop_event = threading.Event()
        loop_event.set()

    try:
        first = True  # Run at least once
        while first or (loop and loop_event.is_set()):
            for pattern_nr in range(num_patterns):
                with nidaqmx.Task() as task:
                    task.do_channels.add_do_chan('Dev1/port0:Dev1/port3', line_grouping=nidaqmx.constants.LineGrouping.CHAN_FOR_ALL_LINES)
                    task.timing.cfg_samp_clk_timing(rate=rate, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=num_samples)
                    if export_clock:
                        task.export_signals.export_signal(nidaqmx.constants.Signal.SAMPLE_CLOCK, '/Dev1/PFI4')
                    else:
                        task.export_signals.export_signal(nidaqmx.constants.Signal.START_TRIGGER, '/Dev1/PFI4')
                    task.write(scan[pattern_nr], auto_start=True)
                    task.wait_until_done(timeout=num_samples/rate + 10)
                    # Between pattern delay
                    #time.sleep(0.001)
            first = False
    except KeyboardInterrupt:
        pass

    if reset_when_done:
        dev.reset_device()


def reset_daq():
    import nidaqmx
    sys = nidaqmx.system.System.local()
    dev = sys.devices['Dev1']
    dev.reset_device()


def main():
    l = np.linspace(0, 2*np.pi, 500)
    #pattern = np.stack([np.sin(l) * 0.5, np.cos(l) * 0.5], axis=1)
    pattern = np.zeros((1000, 2))
    project_patterns(pattern, rate=1000.0, loop = True)


if __name__ == "__main__":
    main()
