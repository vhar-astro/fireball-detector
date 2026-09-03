@echo off
rem UFOCapture appends the completed clip name as the first argument.
"%~dp0fireball-edge\fireball-edge.exe" enqueue --clip-base "%~1" --config "%LOCALAPPDATA%\FireballDetector\edge-config.json"
