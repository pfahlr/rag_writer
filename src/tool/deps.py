from fastapi import HTTPException, Request, status

MAX_IN = 65536


async def size_guard(request: Request) -> None:
    body = await request.body()
    if len(body) > MAX_IN:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Request too large",
        )
