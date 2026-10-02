from fastapi import FastAPI

app = FastAPI(
    title="SkillWatch AI",
    description="AI-based real-time monitoring of training centres",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "name": "SkillWatch AI",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
