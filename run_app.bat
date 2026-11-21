@echo off
REM Double-click this file to start the PD voice screening app.
REM Then open Chrome at http://127.0.0.1:7860  (close this window or press Ctrl+C to stop)

REM go to the project folder (the folder this file is in)
cd /d "%~dp0"

REM switch on the pdvoice conda environment (Anaconda in the user's home folder, else conda on PATH)
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" pdvoice
) else (
    call conda activate pdvoice
)

echo Starting the app... wait for "Running on local URL", then open Chrome at http://127.0.0.1:7860
python app/app.py

REM keep the window open if something goes wrong, so the error can be read
pause
