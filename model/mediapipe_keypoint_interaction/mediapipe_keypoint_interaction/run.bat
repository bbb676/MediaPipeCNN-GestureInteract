@echo off
chcp 65001 >nul
echo MediaPipe Keypoint Gesture Interaction Demo
echo.
echo 1. PPT mode
echo 2. Whiteboard mode
echo 3. Rock Paper Scissors game
echo 4. Debug only, no actions
echo 5. Record custom gesture
set /p choice=Choose mode:
if "%choice%"=="1" python main.py --mode ppt
if "%choice%"=="2" python main.py --mode whiteboard --no-action
if "%choice%"=="3" python main.py --mode game --no-action
if "%choice%"=="4" python main.py --mode debug --no-action
if "%choice%"=="5" (
    set /p gname=Input new gesture label, e.g. call_me:
    python main.py --record %gname% --samples 80 --mode debug --no-action
)
pause
