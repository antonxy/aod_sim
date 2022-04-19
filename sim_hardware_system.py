import sys
sys.path.append('F:/Anton/pco_git')

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
    print(f"Set delay {del_time} {del_timebase}, exposure {exp_time} {exp_timebase}")
    cam.sdk.set_delay_exposure_time(del_time, del_timebase, exp_time, exp_timebase)


def pattern_insert_delay(pattern, pattern_rate_Hz, delay_sec):
    assert len(pattern.shape) == 3

    # We can only insert positive delay. Insert a bit more if it doesn't fit exactly,
    # we can then adjust using camera delay in the other direction.
    num_steps = max(0, math.ceil(delay_sec * pattern_rate_Hz))
    actual_delay = num_steps / pattern_rate_Hz
    first_sample = pattern[:, 0, :]
    repeat_sample = np.tile(first_sample[:, np.newaxis, :], (1, num_steps, 1))
    return np.concatenate([repeat_sample, pattern], axis = 1), actual_delay

class SIMHardwareSystem:
    def __init__(self):
        self.camera = None
        self.camera_exposure = -1;
        self.camera_delay = -1;

        self.multi_frame_acquire = True
        self.multi_frame_num = 0

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

    def configure_camera(self, exposure_time_sec, delay_sec = 0.0, num_frames = 0):
        if exposure_time_sec != self.camera_exposure or delay_sec != self.camera_delay:
            print(f"Set delay {delay_sec} sec, exposure {exposure_time_sec} sec")
            cam_set_delay_exposure_time(self.camera, delay_sec, exposure_time_sec)
            self.camera_exposure = exposure_time_sec
            self.delay_sec = delay_sec

    def project_patterns_and_take_images(self, patterns_deg, pattern_rate_Hz, delay_sec, only_configure=False):
        exposure_time_sec = patterns_deg.shape[1] / pattern_rate_Hz
        num_frames = patterns_deg.shape[0]

        # Since we can only insert fixed steps of delay into the pattern we have
        # to adjust using camera delay after
        print(f"Desired pattern delay {delay_sec} sec")
        patterns_deg_delay, pattern_delay_sec = pattern_insert_delay(patterns_deg, pattern_rate_Hz, delay_sec)
        print(f"Inserted pattern delay {pattern_delay_sec} sec")
        camera_delay_sec = pattern_delay_sec - delay_sec
        print(f"Camera delay {camera_delay_sec} sec")

        quater_sample = 0.25 / pattern_rate_Hz
        self.configure_camera(exposure_time_sec - quater_sample, delay_sec = camera_delay_sec, num_frames = num_frames)

        if self.multi_frame_acquire:
            # Project all patterns without trigger inbetween
            patterns_deg_delay = patterns_deg_delay.reshape(1, -1, 2)
            
            # Somehow the camera needs to get some extra triggers after the desired number of frames
            # Otherwise only e.g. 4 of 9 frames are in the recorder even though oscilloscope shows that
            # 9 frames have been recorded and the 9 frames can also be seen in camware.
            # Very weird..., but thats what this is for
            # ! But it still doesn't work. Now 9 frames are recorded but they are not the first 9 frames,
            # but some random(?) 9 frames of more of them. Is the recorder dropping frames when
            # recording too fast?
            #patterns_deg_delay = np.concatenate([patterns_deg_delay, np.zeros((1, 1000, 2))], axis=1)

        # Start recording. Camera is in trigger mode and will wait for the NI card to start sending the pattern.
        self.camera.record(number_of_images=num_frames, mode='sequence non blocking', memory_mode='camram')
        if only_configure:
            self.camera.stop()
            return

        # Play pattern on the NI card.
        # If we are in multi frame mode send out clock instead of start trigger.
        # This way the camera can record multiple frames, but if the timing is not totally correct AOD and camera might drift apart.
        nidaq_pattern.project_patterns(patterns_deg_delay, pattern_rate_Hz, export_clock = self.multi_frame_acquire)

        # Wait for camera to finish
        t_start = time.time()
        while True:
            running = self.camera.rec.get_status()['is running']
            if not running:
                break
            if (time.time() - t_start) - 0.1 > exposure_time_sec * num_frames:
                self.camera.stop()
                #raise RuntimeError("Camera took too long, probably trigger was lost")
            time.sleep(0.001)

        print("Num images in camera " , self.camera.rec.get_status())
        images, metadatas = self.camera.images()
        if len(images) != num_frames:
            raise RuntimeError("Wrong number of images")
        return np.stack(images)

    def take_widefield_image(self):
        return None

    def project_patterns_looping(self, patterns_deg, pattern_rate_Hz, run_event):
        nidaq_pattern.project_patterns(patterns_deg, pattern_rate_Hz, loop=True, loop_event = run_event)


    def project_patterns_video(self, patterns_deg, pattern_rate_Hz, run_event):
        nidaq_pattern.project_patterns(patterns_deg, pattern_rate_Hz, loop=False, loop_event = run_event, export_clock = True)
