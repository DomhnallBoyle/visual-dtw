from enum import Enum, IntEnum


class Environment(Enum):
    DEVELOPMENT = 1
    PRODUCTION = 2
    TESTING = 3
    RESEARCH = 4


class ListStatus(IntEnum):
    READY = 1
    ARCHIVED = 2
    QUEUED = 3
    POLLED = 4
    UPDATING = 5
