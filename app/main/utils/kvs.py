import pickle

import redis
from main import configuration


class KVS:

    def __init__(self):
        self.r = redis.Redis(host=configuration.REDIS_HOST)

    def keys(self, pattern='*'):
        return self.r.keys(pattern)

    def set(self, k, v):
        self.r.set(k, v)

    def set_obj(self, k, v):
        v = pickle.dumps(v)
        self.set(k, v)

    def get(self, k):
        return self.r.get(k)

    def get_obj(self, k):
        v = self.get(k)

        return pickle.loads(v) if v else None

    def exists(self, k):
        return self.r.exists(k) != 0

    def nested_set(self, k1, k2, v):
        self.r.hset(k1, k2, v)

    def nested_get(self, k1, k2):
        return self.r.hget(k1, k2)

    def clear(self):
        self.r.flushall()
