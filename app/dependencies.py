from fastapi import Header, HTTPException


async def get_token_header(authorization: str = Header(...)):
    prefix, _, token = authorization.partition(" ")
    print(token)
    print(prefix)
    if prefix != "Bearer":
        raise HTTPException(status_code=400, detail="Authorization header invalid")
