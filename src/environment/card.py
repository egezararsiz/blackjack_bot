from .enums import Rank, Suit

class Card:
    """A playing card."""
    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit
    
    def __str__(self):
        return f"{self.rank.name} of {self.suit.name}"
    
    def __repr__(self):
        return self.__str__()

    def get_value(self):
        """Get the blackjack value of the card"""
        if self.rank in [Rank.JACK, Rank.QUEEN, Rank.KING]:
            return 10
        return self.rank.value

    def is_ace(self):
        """Check if the card is an ace"""
        return self.rank == Rank.ACE

    def is_ten_value(self):
        """Check if the card has a value of 10"""
        return self.rank in [Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING]

    def get_count_value(self):
        """Get the card counting value"""
        if self.rank in [Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]:
            return -1
        elif self.rank in [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX]:
            return 1
        return 0  # 7,8,9 are neutral

    def same_color(self, other):
        """Check if two cards are the same color"""
        return self.suit.value % 2 == other.suit.value % 2

    def encode(self):
        """One-hot encode the card for neural network input"""
        index = (self.rank.value - 1) * 4 + self.suit.value
        return index 