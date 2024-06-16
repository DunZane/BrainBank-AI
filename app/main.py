from fastapi import Depends, FastAPI
from app.routers import conversation, document

# app = FastAPI(
#     title="LangChain Server",
#     version="1.0",
#     description="A simple api server using Langchain's Runnable interfaces",
# )
#
# app.include_router(router)

app = FastAPI()

app.include_router(conversation.router)
app.include_router(document.router)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)  # log_level="critical"
