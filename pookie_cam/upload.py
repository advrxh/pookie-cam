import os

import redis

REDIS_URL = os.getenv("REDIS_URL")


class Cache:
    def __init__(self) -> None:
        self.redis = redis.from_url(REDIS_URL)

    def push(self, base64):
        self.redis.lpush("pookies", base64)
        self.redis.ltrim("pookies", 0, 3)
