from typing import List, Optional
from random import Random
from .card import Card
from .enums import Rank, Suit

class Deck:
    """Deck of cards for blackjack."""
    
    def __init__(self, num_decks: int = 8, seed: Optional[int] = None):
        """Initialize with specified number of decks and optional RNG seed."""
        self.num_decks = num_decks
        self.rng = Random(seed)
        self.cards: List[Card] = []
        self.dealt_count = 0
        self.reset()
    
    def reset(self) -> None:
        """Reset and shuffle deck."""
        # Pre-create all cards for efficiency
        self.cards = [
            Card(rank, suit)
            for _ in range(self.num_decks)
            for rank in Rank
            for suit in Suit
        ]
        self.dealt_count = 0
        self.shuffle()
    
    def shuffle(self) -> None:
        """Fast Fisher-Yates shuffle."""
        cards = self.cards
        for i in range(len(cards) - 1, 0, -1):
            j = self.rng.randint(0, i)
            cards[i], cards[j] = cards[j], cards[i]
    
    def deal(self) -> Optional[Card]:
        """Deal one card if available."""
        if self.dealt_count >= len(self.cards):
            return None
        card = self.cards[self.dealt_count]
        self.dealt_count += 1
        return card
    
    def get_penetration(self) -> float:
        """Get current deck penetration."""
        return self.dealt_count / len(self.cards)