import nidaq_pattern
import pco
import time
import numpy as np
import math

def time_to_timebase(time_sec):
    if time_sec <= 4e-3:
        time = int(time_sec * 1e9)
        timebase = 'ns'
    elif time_sec <= 4:
        time = int(time_sec * 1e6)
        timebase = 'us'
    elif time_sec > 4:
        time = int(time_sec * 1e3)
        timebase = 'ms'
    else:
        raise
    return time, timebase


def cam_set_delay_exposure_time(cam, delay_sec, exposure_sec):
    del_time, del_timebase = time_to_timebase(delay_sec)
    exp_time, exp_timebase = time_to_timebase(exposure_sec)
    cam.sdk.set_delay_exposure_time(del_time, del_timebase, exp_time, exp_timebase)


def pattern_insert_delay(pattern, pattern_rate_Hz, delay_sec):
    assert len(pattern.shape) == 3

    # We can only insert positive delay. Insert a bit more if it doesn't fit exactly,
    # we can then adjust using camera delay in the other direction.
    num_steps = max(0, math.ceil(delay_sec * pattern_rate_Hz))
    actual_delay = num_steps * pattern_rate_Hz
    first_sample = pattern[:, 0, :]
    repeat_sample = np.tile(first_sample[:, np.newaxis, :], (1, num_steps, 1))
    return np.concatenate([repeat_sample, pattern], axis = 1), actual_delay

class SIMHardwareSystem:
    def __init__(self):
        self.camera = None
        self.camera_exposure = -1;
        self.camera_delay = -1;

    def connect(self):
        self.camera = pco.Camera(debuglevel='error', interface="Camera Link Silicon Software")
        self.camera.default_configuration()
        self.camera.sdk.set_recorder_submode('sequence')  # It seems this is not set by record
        self.camera.configuration = {
            'roi': (1, 1, 1008, 1008),
            'trigger': 'external exposure start & software trigger',
            #'trigger': 'auto sequence',
            'acquire': 'auto',
        }
        self.configure_camera(1e-3)

    def disconnect(self):
        self.camera.close()
        self.camera = None

    def configure_camera(self, exposure_time_sec, delay_sec = 0.0):
        if exposure_time_sec != self.camera_exposure or delay_sec != self.camera_delay:
            cam_set_delay_exposure_time(self.camera, delay_sec, exposure_time_sec)
            self.camera_exposure = exposure_time_sec
            self.delay_sec = delay_sec

    def project_patterns_and_take_images(self, patterns_deg, pattern_rate_Hz, delay_sec):
        exposure_time_sec = patterns_deg.shape[1] / pattern_rate_Hz

        # Since we can only insert fixed steps of delay into the pattern we have
        # to adjust using camera delay after
        patterns_deg_delay, pattern_delay_sec = pattern_insert_delay(patterns_deg, pattern_rate_Hz, delay_sec)
        self.configure_camera(exposure_time_sec, delay_sec = pattern_delay_sec - delay_sec)

        self.camera.record(number_of_images=patterns_deg.shape[0], mode='sequence non blocking')

        nidaq_pattern.project_patterns(patterns_deg_delay, pattern_rate_Hz)
        while True:
            running = self.camera.rec.get_status()['is running']
            if not running:
                break
            time.sleep(0.001)

        images, metadatas = self.camera.images()
        return np.stack(images)

    def take_widefield_image(self):
        return None

    def project_patterns_looping(self, patterns_deg, pattern_rate_Hz, run_event):
        nidaq_pattern.project_patterns(patterns_deg, pattern_rate_Hz, loop=True, loop_event = run_event)
