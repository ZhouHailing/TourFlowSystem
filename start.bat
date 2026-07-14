@echo off
cd /d "%~dp0"
call ..\tf_venv\Scripts\activate.bat
python manage.py runserver 0.0.0.0:8000
pause
