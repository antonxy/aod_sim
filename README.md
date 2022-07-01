
There are two separate applications: One for dot-scanning LMI and Hex SIM (`sim_gui.py`) and one for line-scanning LMI (`line_lmi_gui.py`).
Both are quite similar.

Start the application in a shell using `python sim_gui.py`.
Using `python sim_gui.py --settings settings.json` a settings file can be loaded on startup (see further down).

On Windows use PowerShell:
```
F: # Switch to F drive
cd Anton/aod_sim # Go to the application folder
python sim_gui.py # start the application
```

## Recording a single stack of images

First thing is usually connecting to the camera. The application hangs while connecting for a few seconds.
Generally while the application is doing something and hangs, don't click on too much stuff, it tends to crash the app.

Define what you want to do: SIM or LMI (Checkboxes), distance between scanning dots, distance in the grating, ...

Set the folder and recording name where you want to save your recording. Each recording is saved in its own subfolder defined by the recording name. The recording notes are saved inside the metadata file with the recording.

If the recording name ends in a number, this number will automatically be incremented after saving. This makes it easy to quickly record a set of slides for example.

You can save your settings to reload later (Very useful in case the app crashes), the settings are also automatically saved with each recording (Useful for later checking what was the exposure time or something).

After changing parameters use "Update Patterns" to update the plots. (Better always do that before recording, it should be done automatically, but I'm not sure it's correct everywhere)

When everything is set record a set of images (`Ctrl+Enter`). You have to save them separately (`Ctrl+S`). There will be a confirmation dialog that they have been saved.

There are Keyboard shortcuts defined for most things, check the menus.

## Recording a video

You can also record a video using camware. For that set all the settings using my app, Use the "set camera settings" command, then disconnect the camera.
If the camera was not connected before "set camera settings" will connect to the camera, set the settings, and disconnect again.

Open Camware, start the recording, and then start use the "Project Pattern Video" command to project the pattern and trigger the camera. The number of recordings is set under "Capture repeats".

## Recording a Z-Stack

To record a z-stack, set the initial stage position and increment in the parameters. Connect to the stage (Home it if not done in Kinesis already) and click on "Capture Z-Stack". The application will freeze (don't click on anything) and will record the z-stack. Each stack position will be saved as a separate recording. The name has to end in a number, this number will be incremented for each stack position.

In the PowerShell window there will be some output showing the current stage position. To end the recording press `Ctrl+C` in the PowerShell window. This aborts the currently running action in the app and the app will unfreeze.
