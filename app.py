from fastapi import Depends, FastAPI
from app.routers import chat, file,ticket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping():
    return {"message": "API server is working!"}


app.include_router(chat.router)
app.include_router(file.router)
app.include_router(ticket.router)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8111)
