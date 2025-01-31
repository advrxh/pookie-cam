import os
from contextlib import asynccontextmanager
import base64

from fastapi import FastAPI, Depends, Response, Request
import aioredis

app = FastAPI()

REDIS_URL = os.getenv("REDIS_URL")
redis = None  # Global Redis instance


async def get_redis(request: Request):
    return request.app.state.redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    yield
    await app.state.redis.close()


app = FastAPI(lifespan=lifespan)


@app.get("/pookie/{i}")
async def get_value(i: int, r: aioredis.Redis = Depends(get_redis)):

    if i >= 0 and i < 4:
        base64_str = await r.lrange("pookies", i, i + 1)
        image_data = base64.b64decode(base64_str[0])
        return Response(content=image_data, media_type="image/png")
    else:
        return "Index out of range"
