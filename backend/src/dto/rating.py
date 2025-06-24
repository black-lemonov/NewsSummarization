from enum import Enum


class RateType(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"


class RateAction(str, Enum):
    ADD = "add"
    REMOVE = "remove"