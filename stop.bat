@echo off
title Stopping LLM Server
cd /d "%~dp0"

echo =============================================
echo   Stopping LLM Server...
echo =============================================
echo.

:: -------------------------------------------------------
:: 1) Stop Python server (server.py)
:: -------------------------------------------------------
echo [1/3] Looking for Python server process...
for /f "tokens=2 delims=," %%A in (
    '2^>nul wmic process where "name='python.exe' and commandline like '%%server.py%%'" get processid /format:csv'
) do (
    echo   Killing Python server (PID: %%A)...
    taskkill /F /PID %%A >nul 2>&1
)

:: Also try with py.exe (some Windows Python launcher setups)
for /f "tokens=2 delims=," %%A in (
    '2^>nul wmic process where "name='py.exe' and commandline like '%%server.py%%'" get processid /format:csv'
) do (
    echo   Killing Python launcher (PID: %%A)...
    taskkill /F /PID %%A >nul 2>&1
)

:: Fallback: kill Python if the WMIC query found nothing
taskkill /F /IM python.exe /FI "WINDOWTITLE eq LLM Server*" >nul 2>&1

echo   Done.
echo.

:: -------------------------------------------------------
:: 2) Stop llama-server.exe (LLM backend on port 8080)
:: -------------------------------------------------------
echo [2/3] Looking for llama-server process...
for /f "tokens=2 delims=," %%A in (
    '2^>nul wmic process where "name='llama-server.exe'" get processid /format:csv'
) do (
    echo   Killing llama-server (PID: %%A)...
    taskkill /F /PID %%A >nul 2>&1
)
echo   Done.
echo.

:: -------------------------------------------------------
:: 3) Free port 8080 and 8443/8000 in case of orphaned sockets
:: -------------------------------------------------------
echo [3/3] Freeing ports (8080, 8000, 8443)...
for %%P in (8080 8000 8443) do (
    for /f "tokens=5" %%A in (
        '2^>nul netstat -ano ^| findstr /R "^ *TCP[^ ]* *0\.0\.0\.0:%%P .* LISTENING"
    ) do (
        echo   Killing process on port %%P (PID: %%A)...
        taskkill /F /PID %%A >nul 2>&1
    )
)
echo   Done.
echo.

:: -------------------------------------------------------
:: Done
:: -------------------------------------------------------
echo =============================================
echo   LLM Server stopped successfully.
echo =============================================
echo.
echo   If you see "[INFO] Shutdown complete." in the
echo   server window, it has closed cleanly.
echo.

pause
