# SkillWatch AI — Diagnostic Update

This copy adds explicit recognition diagnostics. It does not lower the SFace threshold or force a known-person match.

## Start

From this folder, activate your existing virtual environment and run:

```powershell
python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`, connect the camera, and place the enrolled demo person in front of it.

## Exact diagnostic output needed

Run this in a second PowerShell window while the camera is connected:

```powershell
$r = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/recognition/test
$r | ConvertTo-Json -Depth 10
```

Copy the JSON output. The diagnostic includes face quality, `model_row_size`, which recognition branch ran, number of representations successfully decrypted/loaded, feature dimension, best candidate, and raw similarity. It includes no face images or raw face embeddings.

Also capture these endpoints if the test request returns an error:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/recognition/status | ConvertTo-Json -Depth 10
Invoke-RestMethod http://127.0.0.1:8000/api/face/status | ConvertTo-Json -Depth 10
```

## Privacy and contents

The archive omits `.env`, `.git`, Python environments, cached bytecode, and the potentially sensitive recognition event log. The local `data/students.db` and `data/.face_key` are retained so the existing demo enrollment can continue to work. Keep the ZIP private because it contains your local demo database and encryption key.
