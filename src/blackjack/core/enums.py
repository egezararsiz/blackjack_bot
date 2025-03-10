from enum import Enum

class Rank(Enum):
    """Card ranks with their blackjack values."""
    ACE = 1     # Special handling in Hand class for 1/11
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
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3

class Action(Enum):
    """Player actions that can be taken during play."""
    HIT = 0
    STAND = 1
    DOUBLE = 2
    SPLIT = 3
    INSURANCE = 4

class BetType(Enum):
    """Types of bets available at the start of a hand."""
    MAIN = 0
    PERFECT_PAIRS = 1
    TWENTYONE_PLUS_THREE = 2
    INSURANCE = 3

class HandStatus(Enum):
    """Current status/outcome of a hand after actions have been taken.
    Represents the state of the hand rather than available actions."""
    ACTIVE = 0     # Hand is still active, waiting for player decisions
    BUST = 1        # Hand value exceeded 21
    STAND = 2       # Player has finished with this hand
    BLACKJACK = 3   # Natural blackjack (Ace + 10-value card)
    SPLIT = 4       # Hand has been split into two separate hands
    DOUBLE = 5      # Hand has been doubled down

class PairType(Enum):
    """Types of pairs for Perfect Pairs bet."""
    NONE = 0
    MIXED = 1  # Different colors (♠♥ or ♣♦)
    COLORED = 2  # Same color, different suit (♠♣ or ♥♦)
    PERFECT = 3  # Same suit (♠♠, ♥♥, ♣♣, ♦♦)

class PokerHand(Enum):
    """Poker hand rankings for 21+3 bet."""
    NONE = 0
    FLUSH = 1
    STRAIGHT = 2
    THREE_OF_A_KIND = 3
    STRAIGHT_FLUSH = 4
    SUITED_TRIPS = 5