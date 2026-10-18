Write-Host "Starting SkillWatch AI backend..."
if (Test-Path .venv\Scripts\Activate.ps1) {
    . .venv\Scripts\Activate.ps1
}
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
