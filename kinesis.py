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

    def connect(self):
        DeviceManagerCLI.BuildDeviceList()
        self.device = TCubeDCServo.CreateTCubeDCServo(self.serial)
        print(self.device)
        self.device.Connect(self.serial)
        deviceInfo = self.device.GetDeviceInfo()
        print(deviceInfo.Name, '  ', deviceInfo.SerialNumber)
        self.device.WaitForSettingsInitialized(5000)
        self.device.StartPolling(100)
        time.sleep(0.5)
        self.device.EnableDevice()
        time.sleep(0.5)
        self.device.LoadMotorConfiguration(self.serial)
       
    def disconnect(self):
        self.device.StopPolling()
        self.device.Disconnect(True)

    def home(self):
        self.device.Home(60000)

    def move_to(self, position_mm):
        self.device.MoveTo(Decimal(position_mm), 60000)
    
    def position_mm(self):
        return float(str(self.device.Position))

if __name__ == '__main__':
    s = Stage(stage_serial_number)
    s.connect()
    #s.home()
    print(s.position_mm())
    s.move_to(2.0)
    print(s.position_mm())