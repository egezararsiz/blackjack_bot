from enum import Enum, auto

class Rank(Enum):
    """Card ranks."""
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 10
    QUEEN = 10
    KING = 10

class Suit(Enum):
    """Card suits."""
    HEARTS = auto()
    DIAMONDS = auto()
    CLUBS = auto()
    SPADES = auto()

class Action(Enum):
    """Player actions."""
    HIT = 0
    STAND = 1
    DOUBLE = 2
    SPLIT = 3
    INSURANCE = 4 