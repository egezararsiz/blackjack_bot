from dataclasses import dataclass
from .enums import Rank, Suit

@dataclass(frozen=True)
class Card:
    """Immutable playing card."""
    rank: Rank
    suit: Suit
    
    def get_value(self) -> int:
        """Get the blackjack value of the card.
        
        Note: Aces are returned as 1. The decision to use it as 11
        should be handled by the hand evaluation logic, since it depends
        on the total value of all cards in the hand."""
        return self.rank.value

    def is_ace(self) -> bool:
        """Check if the card is an ace."""
        return self.rank == Rank.ACE
    
    def is_ten_value(self) -> bool:
        """Check if the card is a ten-value card."""
        return self.rank.value == 10