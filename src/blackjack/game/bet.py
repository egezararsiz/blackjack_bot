from dataclasses import dataclass
from typing import Tuple, List

from blackjack.core.card import Card, Rank
from ..core.hand import Hand
from ..core.rules import BlackjackRules
from ..core.enums import BetType, HandStatus, PairType, PokerHand

@dataclass
class Bet:
    """
    Represents bets for a hand.
    Responsible for tracking bet amounts for main and side bets.
    """
    main: float = 0.0
    perfect_pairs: float = 0.0
    twentyone_plus_three: float = 0.0
    insurance: float = 0.0
    
    def place_bet(self, bet_type: BetType, amount: float) -> None:
        """Place a bet of specified type and amount."""
        if bet_type == BetType.MAIN:
            self.main = amount
        elif bet_type == BetType.PERFECT_PAIRS:
            self.perfect_pairs = amount
        elif bet_type == BetType.TWENTYONE_PLUS_THREE:
            self.twentyone_plus_three = amount
        elif bet_type == BetType.INSURANCE:
            self.insurance = amount
    
    def double_bet(self) -> None:
        """Double the main bet."""
        self.main *= 2
    
    def get_total(self) -> float:
        """Get total amount bet."""
        return (self.main + self.perfect_pairs + 
                self.twentyone_plus_three + self.insurance)

@dataclass
class BetManager:
    """
    Manages betting operations.
    Responsible for:
    - Validating bet amounts
    - Calculating payouts using rules multipliers
    - Evaluating side bet conditions
    """
    rules: BlackjackRules
    min_bet: float = 1.0
    max_bet: float = 100.0
    
    def get_valid_bet_range(self, bet_type: BetType) -> Tuple[float, float]:
        """Get valid bet range for specified bet type."""
        if bet_type == BetType.MAIN:
            return self.min_bet, self.max_bet
        elif bet_type == BetType.INSURANCE:
            return 0.0, self.max_bet / 2
        else:  # Side bets
            return 0.0, self.max_bet
    
    def calculate_payout(self, bet: Bet, hand: Hand, dealer_hand: Hand) -> float:
        """Calculate total payout for all bets."""
        total_payout = 0.0
        
        # Main bet payout
        if hand.status == HandStatus.BLACKJACK:
            dealer_blackjack = dealer_hand.status == HandStatus.BLACKJACK
            multiplier = self.rules.get_payout_multiplier(hand, dealer_blackjack)
            total_payout += bet.main * (1 + multiplier)
        elif hand.status == HandStatus.BUST:
            total_payout += 0  # Player loses
        elif dealer_hand.status == HandStatus.BUST:
            total_payout += bet.main * 2  # Player wins
        else:
            # Compare hand values
            if hand.get_value() > dealer_hand.get_value():
                total_payout += bet.main * 2  # Player wins
            elif hand.get_value() == dealer_hand.get_value():
                total_payout += bet.main  # Push
                
        return total_payout
    
    def calculate_insurance_payout(self, bet: Bet, dealer_blackjack: bool) -> float:
        """Calculate insurance payout."""
        if not bet.insurance:
            return 0.0
        multiplier = self.rules.get_insurance_payout(dealer_blackjack)
        return bet.insurance * (1 + multiplier)
    
    def calculate_perfect_pairs_payout(self, bet: Bet, hand: Hand) -> float:
        """Calculate Perfect Pairs side bet payout."""
        if not bet.perfect_pairs or len(hand.cards) != 2:
            return 0.0
            
        # Determine pair type
        if not hand.is_pair():
            return 0.0
            
        # Get pair type and multiplier from rules
        pair_type = self._get_pair_type(hand.cards[0], hand.cards[1])
        multiplier = self.rules.get_perfect_pairs_multiplier(pair_type)
        return bet.perfect_pairs * (1 + multiplier)
    
    def calculate_21plus3_payout(self, bet: Bet, player_hand: Hand, 
                                dealer_upcard: Card) -> float:
        """Calculate 21+3 side bet payout."""
        if not bet.twentyone_plus_three or len(player_hand.cards) != 2:
            return 0.0
            
        # Create three-card hand for poker evaluation
        cards = player_hand.cards[:2] + [dealer_upcard]
        poker_hand = self._evaluate_poker_hand(cards)
        multiplier = self.rules.get_21plus3_multiplier(poker_hand)
        return bet.twentyone_plus_three * (1 + multiplier)
    
    def _get_pair_type(self, card1: Card, card2: Card) -> PairType:
        """Determine pair type for Perfect Pairs."""
        if card1.rank != card2.rank:
            return PairType.NONE
        if card1.suit == card2.suit:
            return PairType.PERFECT
        if card1.is_red() == card2.is_red():
            return PairType.COLORED
        return PairType.MIXED
    
    def _evaluate_poker_hand(self, cards: List[Card]) -> PokerHand:
        """Evaluate three-card poker hand for 21+3."""
        if self._is_suited_trips(cards):
            return PokerHand.SUITED_TRIPS
        if self._is_straight_flush(cards):
            return PokerHand.STRAIGHT_FLUSH
        if self._is_three_of_a_kind(cards):
            return PokerHand.THREE_OF_A_KIND
        if self._is_straight(cards):
            return PokerHand.STRAIGHT
        if self._is_flush(cards):
            return PokerHand.FLUSH
        return PokerHand.NONE
    
    def _is_suited_trips(self, cards: List[Card]) -> bool:
        """Check for suited three of a kind (highest paying hand)."""
        return (self._is_three_of_a_kind(cards) and 
                self._is_flush(cards))
    
    def _is_straight_flush(self, cards: List[Card]) -> bool:
        """Check for straight flush (second highest paying hand)."""
        return self._is_straight(cards) and self._is_flush(cards)
    
    def _is_three_of_a_kind(self, cards: List[Card]) -> bool:
        """
        Check for three of a kind.
        More efficient than checking all against first.
        """
        return (cards[0].rank == cards[1].rank == cards[2].rank)
    
    def _is_straight(self, cards: List[Card]) -> bool:
        """
        Check for straight using card ranks.
        Valid straights:
        - Regular sequence (e.g., 2-3-4, 9-10-J)
        - Q-K-A: Ace acts as high card after King
        """
        ranks = [card.rank for card in cards]
        
        # Get the ordinal positions in the enum (1-based index)
        # This gives us proper ordering: A(1), 2(2), ..., 10(10), J(11), Q(12), K(13)
        positions = [list(Rank).index(r) + 1 for r in ranks]
        positions.sort()
        
        # Check for Q-K-A case (positions would be 1,12,13)
        if positions == [1, 12, 13]:
            return True
            
        # For regular straights, check if positions are consecutive
        return (positions[1] == positions[0] + 1 and 
                positions[2] == positions[1] + 1)
    
    def _is_flush(self, cards: List[Card]) -> bool:
        """Check for flush (all same suit)."""
        return (cards[0].suit == cards[1].suit == cards[2].suit) 