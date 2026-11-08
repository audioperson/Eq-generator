<img width="1512" height="982" alt="Bildschirmfoto 2026-11-01 um 12 48 12" src="https://github.com/user-attachments/assets/599d8124-1778-48bc-90c3-c42e3f77c84b" />
For my project, I wrote a program that generates an EQ setting based on a comparison between a reference audio track and another track (e.g., one that has been recorded). This makes it possible to calibrate a PA system for flat (or other presets) frequency response. Currently, the program only runs on Mac, and the audio device must be selected manually or configured via Audio MIDI Setup.(that will change in v0.5) Of curse I am working on a good settings menu where you can just select your audio device and the channels so it more friendly for the user. The future versions will generate not an flat eq but an eq that will optimize (e.g. voice) for your pa and room, this thing works with some reference based on studies and experiments and combines that with the eq generation.
In the new v0.5 is a room analysis tab in the settings menu. To activate it, press in settings then go in the sidebar to room and select start room analysis. This will calculate the responding of the room to a loud boom and will also calculate the time the sound "stays" in the room in different frequencies. This will take 30s of sound.

It can generate an eq with live audio (an audio interface) or with already recorded files. It will generate a graphic eq and eq values to enter in the eq (on your mixer). To open it download the program from the release unpack it and start it. now you need to set the mode, for already recorded files use the recorded mode. to use that click on the button and select the files. For the live mode you can use the test mode with recorded files or theoretically use it live (v0.5 or higher)


I want to Programm this because I work with sound systems and mix bands and the tools that exists are expensive and are not exactly something I need so I started programming a tool for It, in the v0.3 or higher, is an graphical user interface for better usability. And with this project I have something to do and also learn how to calculate with audio.

the 0.4 version now is less laggy and has an anti noise via kross correlation and an upgraded time difference correction (via peaks in the frequency) It also has an settings menu to select the audio devices and shows errors.

The project has (in v0.3 and 0.4) to files first vergleich.py this is the "old" code, wich has the main calculation in it, it is used in gui.py as a library. gui.py features the GUI and also some calculation, wich was later added. The v0.5 will also have test_room.py wich calculates the room.

the new code is in the v0.5 folder





st
