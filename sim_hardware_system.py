import nidaq_pattern
import pco

class SIMHardwareSystem:
    def __init__(self):
        self.camera = None
        self.camera_exposure = -1;

    def connect(self):
        self.camera = pco.Camera(debuglevel='error', interface="Camera Link Silicon Software")
        self.camera.default_configuration()
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

    def configure_camera(self, exposure_time_sec):
        if exposure_time_sec != self.camera_exposure:
            self.camera.configuration = {
                'exposure time': exposure_time_sec,
            }
            self.camera_exposure = exposure_time_sec

    def project_patterns_and_take_images(self, patterns_deg, pattern_rate_Hz):
        exposure_time_sec = patterns_deg.shape[1] / pattern_rate_Hz
        self.configure_camera(exposure_time_sec)

        self.camera.record(number_of_images=7, mode='sequence non blocking')

        nidaq_pattern.project_patterns(patterns_deg, pattern_rate_Hz)
        while True:
            running = self.camera.rec.get_status()['is running']
            if not running:
                break
            time.sleep(0.001)

        images, metadatas = self.camera.images()
        return np.stack(images)

    def project_patterns_looping(self, patterns_deg, pattern_rate_Hz, run_event):
        nidaq_pattern.project_patterns(patterns_deg, pattern_rate_Hz, loop=True, loop_event = run_event)
