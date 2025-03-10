from typing import List, Tuple
from dataclasses import dataclass, field
from .card import Card
from .enums import HandStatus

@dataclass
class Hand:
    """
    Represents a blackjack hand.
    Responsible for:
    - Managing cards and hand state
    - Calculating hand values
    - Tracking split status
    """
    cards: List[Card] = field(default_factory=list)
    status: HandStatus = field(default=HandStatus.ACTIVE)
    split_from_aces: bool = field(default=False)
    
    def __post_init__(self):
        """Initialize hand status."""
        # Check for natural blackjack in initial hand
        if len(self.cards) == 2 and self._calculate_value() == 21:
            self.status = HandStatus.BLACKJACK
    
    def add_card(self, card: Card) -> None:
        """Add card to hand and update status."""
        self.cards.append(card)
        self._update_status_after_hit()
    
    def _update_status_after_hit(self) -> None:
        """Update hand status after hitting."""
        if self._calculate_value() > 21:
            self.status = HandStatus.BUST
        elif self.split_from_aces and len(self.cards) > 1:
            self.status = HandStatus.STAND
    
    def get_value(self) -> int:
        """Get optimal hand value."""
        return self._calculate_value()
    
    def _calculate_value(self) -> int:
        """Calculate optimal hand value."""
        non_ace_total, ace_count = self._count_values()
        return self._optimize_ace_values(non_ace_total, ace_count)
    
    def _count_values(self) -> Tuple[int, int]:
        """Count non-ace total and number of aces."""
        non_ace_total = 0
        ace_count = 0
        
        for card in self.cards:
            if card.is_ace():
                ace_count += 1
            else:
                non_ace_total += card.get_value()
                
        return non_ace_total, ace_count
    
    def _optimize_ace_values(self, non_ace_total: int, ace_count: int) -> int:
        """Determine optimal values for aces."""
        if ace_count == 0:
            return non_ace_total
            
        # Try using one ace as 11 if possible
        if non_ace_total + 11 + (ace_count - 1) <= 21:
            return non_ace_total + 11 + (ace_count - 1)
            
        # All aces count as 1
        return non_ace_total + ace_count
    
    def is_soft(self) -> bool:
        """Check if hand is soft (contains an ace counted as 11)."""
        non_ace_total, ace_count = self._count_values()
        if ace_count == 0:
            return False
        return non_ace_total + 11 + (ace_count - 1) <= 21
    
    def is_pair(self) -> bool:
        """Check if hand is a pair."""
        return len(self.cards) == 2 and self.cards[0].rank == self.cards[1].rank
    
    def split(self) -> 'Hand':
        """Split hand into two hands."""
        split_card = self.cards.pop()
        new_hand = Hand(
            cards=[split_card],
            split_from_aces=self.cards[0].is_ace()
        )
        self.status = HandStatus.SPLIT
        return new_hand
    
    def double(self) -> None:
        """Mark hand as doubled."""
        self.status = HandStatus.DOUBLE
    
    def stand(self) -> None:
        """Mark hand as standing."""
        if self.status == HandStatus.ACTIVE:
            self.status = HandStatus.STAND