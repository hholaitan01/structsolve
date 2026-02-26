@echo off
echo =============================================
echo   StructSolve — First-Time Setup
echo =============================================
echo.

echo [1/4] Creating virtual environment...
python -m venv .venv

echo [2/4] Activating virtual environment...
call .venv\Scripts\activate

echo [3/4] Installing dependencies...
pip install -r requirements.txt

echo [4/4] Done! Starting StructSolve...
echo.
echo   Open your browser at: http://localhost:8501
echo.
streamlit run combine.py
