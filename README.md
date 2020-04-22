# CMLS HW1 - Music Emotion Recognition

## Deadlines

- **May 17th:** groups are expected to upload on Beep a detailed report on the project (max 8 pages) on a prescribed Beep folder, to be communicated shortly. The reports will be made available to the other students for the evaluation. Due to the GDPR regulation, please do not include in the report the names of the group components, but only the personal codes. You can also choose a name of your preference for the group, which is for now only a number (1...20).

  Important note: based on the report, the students evaluating the projects will have to assess: a) the originality of the proposed solution and b) the quality of the implementation, for a total of 8/3 (2.66) points.

- **May 25th:** public presentation of the HWs during the usual class time, using a PPT presentation. Due to the limited time, all presentations are expected must last no more than 5 minutes.

## Document

You can use any method of your choice to compile the LaTeX source, the file to be compiled is `main.tex`.
Built files will be ignored by `git`, please don't track them.

## Project

Some steps are required to deal with the project. For convenience I created some scripts that can just double-clicked to be executed, but in case of any issue you can open them with a text editor to see the commands and try to execute them by yourself (make sure to `cd` inside the `project/` directory first).

1. Make sure you have `python` installed.
2. *Setup the virtual environment:* execute `setup.sh` (on Linux/MacOS) or `setup.bat` (on Windows). This will install `pipenv` and install the needed dependencies.
3. *Open Juypiter Notebook:* execute `editor.sh` (on Linux/MacOS) or `editor.bat` (on Windows). This will enable some extensions and start Jupyter's server.
4. Profit. Next times, if you don't encounter dependency issues, you only need the 3rd step.
