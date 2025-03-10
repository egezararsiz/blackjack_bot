from dataclasses import dataclass
from typing import Dict
from .enums import PairType, HandStatus, PokerHand
from .hand import Hand

@dataclass(frozen=True)
class BlackjackRules:
    """
    Immutable container for blackjack rules and payout multipliers.
    Single source of truth for:
    - Game rules and constraints
    - Payout multipliers for all bet types
    - Action validation
    """
    # Basic rules
    num_decks: int = 8
    dealer_hits_soft_17: bool = False  # Dealer stands on all 17s
    blackjack_pays: float = 1.5  # 3:2
    insurance_payout: float = 2.0  # 2:1
    
    # Splitting rules
    max_split_hands: int = 2  # Only one split allowed
    can_hit_split_aces: bool = False  # One card only to split aces
    can_resplit_aces: bool = False  # No resplitting
    can_double_after_split: bool = False  # No double after split
    split_aces_pay_bonus: bool = False  # Split aces pay 1:1
    
    # Double down rules
    can_double_any_two: bool = True  # Can double on any first two cards
    min_double_total: int = 0  # No minimum total required
    max_double_total: int = 21  # No maximum total restriction
    
    # Dealer rules
    dealer_peeks: bool = True  # Vegas rules - dealer peeks for blackjack
    
    # Side bet payouts
    perfect_pairs_payouts: Dict[PairType, float] = None
    twentyone_plus_three_payouts: Dict[PokerHand, float] = None
    
    def __post_init__(self):
        """Initialize immutable side bet payouts."""
        # Perfect Pairs payouts
        object.__setattr__(self, 'perfect_pairs_payouts', {
            PairType.PERFECT: 25.0,  # Same suit (♠♠) pays 25:1
            PairType.COLORED: 12.0,  # Same color (♠♣) pays 12:1
            PairType.MIXED: 6.0,     # Mixed color (♠♥) pays 6:1
            PairType.NONE: 0.0
        })
        
        # 21+3 payouts
        object.__setattr__(self, 'twentyone_plus_three_payouts', {
            PokerHand.SUITED_TRIPS: 100.0,    # Three of a kind, same suit
            PokerHand.STRAIGHT_FLUSH: 40.0,   # Three consecutive cards, same suit
            PokerHand.THREE_OF_A_KIND: 30.0,  # Three of a kind, mixed suits
            PokerHand.STRAIGHT: 10.0,         # Three consecutive cards, mixed suits
            PokerHand.FLUSH: 5.0,            # Three cards of same suit
            PokerHand.NONE: 0.0
        })
    
    @property
    def min_cards_before_shuffle(self) -> int:
        """Minimum cards required before forced shuffle."""
        total_cards = self.num_decks * 52
        return int(total_cards * 0.5)  # 50% penetration for online play
    
    def can_hit(self, hand: 'Hand') -> bool:
        """Check if hand can receive another card."""
        if hand.status != HandStatus.ACTIVE:
            return False
        if hand.split_from_aces and len(hand.cards) >= 1:
            return False
        return True
    
    def can_split(self, hand: 'Hand', num_hands: int) -> bool:
        """Check if hand can be split."""
        if hand.status != HandStatus.ACTIVE:
            return False
        if len(hand.cards) != 2:
            return False
        if not hand.is_pair():
            return False
        if num_hands >= self.max_split_hands:
            return False
        return True
    
    def can_double(self, hand: 'Hand', is_after_split: bool = False) -> bool:
        """Check if hand can be doubled."""
        if hand.status != HandStatus.ACTIVE:
            return False
        if len(hand.cards) != 2:
            return False
        if is_after_split and not self.can_double_after_split:
            return False
        return True
    
    def get_payout_multiplier(self, hand: 'Hand', dealer_blackjack: bool = False) -> float:
        """Get payout multiplier for a winning hand."""
        if hand.status == HandStatus.BLACKJACK:
            if dealer_blackjack:
                return 1.0  # Push
            if hand.split_from_aces and not self.split_aces_pay_bonus:
                return 1.0  # Split aces pay 1:1
            return self.blackjack_pays
        return 1.0  # Regular win
    
    def get_insurance_payout(self, dealer_blackjack: bool) -> float:
        """Get insurance payout multiplier."""
        return self.insurance_payout if dealer_blackjack else 0.0
    
    def should_dealer_hit(self, dealer_value: int, is_soft: bool) -> bool:
        """Determine if dealer should hit based on rules."""
        if dealer_value > 17:
            return False
        if dealer_value < 17:
            return True
        return self.dealer_hits_soft_17 and is_soft
    
    def get_perfect_pairs_multiplier(self, pair_type: PairType) -> float:
        """Get multiplier for Perfect Pairs bet."""
        return self.perfect_pairs_payouts[pair_type]
    
    def get_21plus3_multiplier(self, poker_hand: PokerHand) -> float:
        """Get multiplier for 21+3 bet."""
        return self.twentyone_plus_three_payouts[poker_hand] 