import clr
import sys
import time

from System import String
from System import Decimal
from System.Collections import *

# constants
sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")
stage_serial_number = '83841611'

# add .net reference and import so python can see .net
clr.AddReference("Thorlabs.MotionControl.Controls")
import Thorlabs.MotionControl.Controls

# print(Thorlabs.MotionControl.Controls.__doc__)

# Add references so Python can see .Net
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
clr.AddReference("Thorlabs.MotionControl.TCube.DCServoCLI")
from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.TCube.DCServoCLI import *;


class Stage(object):
    def __init__(self, serial):
        self.serial = serial
        self.device = None

    def init(self):
        self.device = TCubeDCServo.CreateTCubeDCServo(self.serial)
        self.device.Connect(self.serial)
        deviceInfo = self.device.GetDeviceInfo()
        print(deviceInfo.Name, '  ', deviceInfo.SerialNumber)
        self.device.WaitForSettingsInitialized(5000)
        self.device.StartPolling(100)
        time.sleep(0.5)
        self.device.EnableDevice()
        time.sleep(0.5)
        self.device.LoadMotorConfiguration(self.serial)

    def home(self):
        self.device.Home(60000)

    def move_to(self, position):
        self.device.MoveTo(Decimal(position), 60000)
