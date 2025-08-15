@echo off
REM Installation script for Agent Core (Windows)

echo =========================================
echo      Agent Core Installation (Windows)
echo =========================================
echo.

REM Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    exit /b 1
)

echo Python is installed
echo.

REM Parse arguments
set MODE=%1
if "%MODE%"=="" set MODE=basic

if "%MODE%"=="basic" (
    echo Installing basic requirements...
    pip install --user -r requirements.txt
    goto :done
)

if "%MODE%"=="dev" (
    echo Installing development requirements...
    pip install --user -r requirements-dev.txt
    echo.
    echo Setting up pre-commit hooks...
    pre-commit install
    goto :done
)

if "%MODE%"=="docker" (
    echo Checking Docker installation...
    docker --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Error: Docker is not installed
        echo Please install Docker Desktop from: https://docs.docker.com/desktop/windows/install/
        exit /b 1
    )
    
    echo Docker is installed
    echo.
    echo Building Docker images...
    docker-compose -f docker-compose.api.yml build
    goto :done
)

if "%MODE%"=="full" (
    echo Full installation...
    
    REM Install Python requirements
    echo Installing Python packages...
    pip install --user -r requirements.txt
    pip install --user -r requirements-dev.txt
    
    REM Create necessary directories
    echo.
    echo Creating directories...
    if not exist "data" mkdir data
    if not exist "logs" mkdir logs
    
    REM Copy environment template if not exists
    if not exist ".env" (
        echo Creating .env file from template...
        copy config\.env.example .env
        echo Please edit .env file with your API keys
    )
    goto :done
)

if "%MODE%"=="test" (
    echo Running installation tests...
    
    REM Test imports
    python -c "import fastapi; print('OK: FastAPI installed')"
    python -c "import uvicorn; print('OK: Uvicorn installed')"
    python -c "import aiohttp; print('OK: Aiohttp installed')"
    python -c "import redis; print('OK: Redis-py installed')"
    python -c "import openai; print('OK: OpenAI installed')"
    
    REM Test API startup
    echo.
    echo Testing API startup...
    python -c "from api import app; print('OK: API imports successfully')"
    goto :done
)

echo Usage: install.bat [mode]
echo.
echo Modes:
echo   basic  - Install basic requirements only (default)
echo   dev    - Install development requirements
echo   docker - Build Docker images
echo   full   - Full installation
echo   test   - Test installation
echo.
echo Examples:
echo   install.bat basic
echo   install.bat dev
echo   install.bat full
exit /b 1

:done
echo.
echo =========================================
echo      Installation Complete!
echo =========================================
echo.
echo Next steps:
echo 1. Set your OPENROUTER_API_KEY in .env file
echo 2. Run the API: python api.py
echo 3. Visit docs: http://localhost:8000/docs
echo.