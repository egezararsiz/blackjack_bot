from typing import List
from ..core.card import Card
from ..core.enums import PairType, PokerHand, Rank

def evaluate_perfect_pairs(cards: List[Card]) -> PairType:
    """Evaluate Perfect Pairs side bet."""
    if len(cards) != 2 or cards[0].rank != cards[1].rank:
        return PairType.NONE
        
    # Perfect pair (same suit)
    if cards[0].suit == cards[1].suit:
        return PairType.PERFECT
        
    # Colored pair (same color)
    if (cards[0].suit.value < 2) == (cards[1].suit.value < 2):
        return PairType.COLORED
        
    # Mixed pair (different colors)
    return PairType.MIXED

def evaluate_21plus3(player_cards: List[Card], dealer_card: Card) -> PokerHand:
    """Evaluate 21+3 side bet."""
    if len(player_cards) != 2:
        return PokerHand.NONE
        
    cards = player_cards + [dealer_card]
    
    # Check for three of a kind first (most valuable)
    if all(c.rank == cards[0].rank for c in cards):
        if all(c.suit == cards[0].suit for c in cards):
            return PokerHand.SUITED_TRIPS
        return PokerHand.THREE_OF_A_KIND
    
    # Check for flush
    is_flush = all(c.suit == cards[0].suit for c in cards)
    
    # Check for straight
    is_straight = _is_straight(cards)
    
    if is_flush and is_straight:
        return PokerHand.STRAIGHT_FLUSH
    elif is_straight:
        return PokerHand.STRAIGHT
    elif is_flush:
        return PokerHand.FLUSH
        
    return PokerHand.NONE

def _is_straight(cards: List[Card]) -> bool:
    """
    Check if cards form a straight.
    Uses actual ranks instead of card values to handle 10-J-Q-K sequences.
    """
    # Get unique ranks in ascending order
    unique_ranks = sorted(set(c.rank for c in cards), key=lambda r: r.value)
    
    # Need exactly 3 different ranks
    if len(unique_ranks) != 3:
        return False
        
    # Special case: Ace-2-3
    if (unique_ranks[0] == Rank.ACE and 
        unique_ranks[1] == Rank.TWO and 
        unique_ranks[2] == Rank.THREE):
        return True
        
    # Special case: Queen-King-Ace
    if (unique_ranks[0] == Rank.QUEEN and 
        unique_ranks[1] == Rank.KING and 
        unique_ranks[2] == Rank.ACE):
        return True
        
    # Get the ordinal positions of the ranks (1-13)
    positions = [r.value if r != Rank.ACE else 14 for r in unique_ranks]
    
    # Normal case: three consecutive ranks
    return positions[2] - positions[0] == 2 and positions[1] - positions[0] == 1 