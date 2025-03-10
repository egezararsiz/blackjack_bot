from dataclasses import dataclass, field
from typing import List
from ..core.card import Card
from ..core.hand import Hand
from ..core.rules import BlackjackRules

@dataclass
class Dealer:
    """
    Represents the dealer in blackjack.
    Handles dealer-specific rules and logic.
    """
    hand: Hand = field(default_factory=Hand)
    rules: BlackjackRules = field(default_factory=BlackjackRules)
    
    def reset(self) -> None:
        """Reset dealer's hand."""
        self.hand = Hand()
    
    def add_card(self, card: Card) -> None:
        """Add a card to dealer's hand."""
        self.hand.add_card(card)
    
    def should_hit(self) -> bool:
        """
        Determine if dealer should hit according to rules.
        Dealer must hit on soft 17 if rules specify.
        """
        value = self.hand.get_value()
        if value > 17:
            return False
        if value < 17:
            return True
        # On exactly 17, hit only if it's soft and rules require it
        return self.rules.dealer_hits_soft_17 and self.hand.is_soft()
    
    def get_upcard(self) -> Card:
        """Get dealer's face-up card."""
        if not self.hand.cards:
            raise ValueError("Dealer has no cards")
        return self.hand.cards[0]
    
    def has_blackjack(self) -> bool:
        """Check if dealer has blackjack."""
        return self.hand.is_blackjack()
    
    def is_bust(self) -> bool:
        """Check if dealer is bust."""
        return self.hand.get_value() > 21
    
    def get_value(self) -> int:
        """Get dealer's hand value."""
        return self.hand.get_value()
    
    def __str__(self) -> str:
        """String representation showing only upcard if hand not complete."""
        if len(self.hand.cards) == 2 and not self.has_blackjack():
            return f"Dealer showing: {self.get_upcard()}"
        return f"Dealer hand: {self.hand}" 