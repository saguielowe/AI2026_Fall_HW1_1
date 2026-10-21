@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
if not exist ".venv\Scripts\python.exe" (
  echo Missing project Python environment.
  exit /b 1
)
if "%~1"=="calculation" goto calculation
if "%~1"=="sudoku" goto sudoku
if "%~1"=="mcts" goto mcts
if "%~1"=="check" goto check
if "%~1"=="build" goto build
echo Usage: start-homework.cmd calculation ^| sudoku ^| mcts ^| check ^| build
exit /b 0
:calculation
".venv\Scripts\python.exe" "code\calculation\visualizer.py" --port 18767
exit /b %errorlevel%
:sudoku
".venv\Scripts\python.exe" "code\sudoku\visualizer.py"
exit /b %errorlevel%
:mcts
".venv\Scripts\python.exe" "code\mcts\visualizer.py"
exit /b %errorlevel%
:check
".venv\Scripts\python.exe" ".environment\check_environment.py"
exit /b %errorlevel%
:build
".venv\Scripts\python.exe" ".environment\build_go.py"
exit /b %errorlevel%
