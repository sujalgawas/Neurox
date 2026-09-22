from fastapi import FastAPI, Depends
from app.config import Settings,get_settings

app = FastAPI()

@app.get("/")
def health():
    return {"message":"backend is working"}

@app.get("/test_env")
def test_env(env: Settings = Depends(get_settings)):
    return {"message": env.ALPACA_BASE_URL}