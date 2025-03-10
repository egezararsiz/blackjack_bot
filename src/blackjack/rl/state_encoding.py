from typing import List
from ..core.card import Card
from ..core.deck import Deck
from ..core.enums import Rank, Suit

class StateEncoder:
    """Handles encoding of game state for RL agent."""
    
    @staticmethod
    def encode_card(card: Card) -> int:
        """Convert card to one-hot index."""
        return (card.rank.value - 1) * 4 + card.suit.value
    
    @staticmethod
    def encode_deck_state(deck: Deck) -> List[int]:
        """Encode remaining cards in deck."""
        counts = [0] * 52  # 13 ranks * 4 suits
        for i in range(deck.dealt_count, len(deck.cards)):
            card = deck.cards[i]
            counts[StateEncoder.encode_card(card)] += 1
        return counts
    
    @staticmethod
    def decode_card_index(index: int) -> tuple[Rank, Suit]:
        """Convert one-hot index back to rank and suit."""
        rank_value = (index // 4) + 1
        suit_value = index % 4
        return Rank(rank_value), Suit(suit_value) 