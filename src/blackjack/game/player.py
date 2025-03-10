from dataclasses import dataclass, field
from typing import List, Optional
from ..core.hand import Hand
from ..core.card import Card
from ..core.rules import BlackjackRules
from ..core.enums import Action, HandStatus

@dataclass
class Player:
    """
    Represents a player in blackjack.
    Manages multiple hands and available actions.
    """
    bankroll: float
    hands: List[Hand] = field(default_factory=list)
    current_hand_index: int = 0
    rules: BlackjackRules = field(default_factory=BlackjackRules)
    
    def __post_init__(self):
        """Initialize with one empty hand."""
        self.reset()
    
    def reset(self) -> None:
        """Reset player's hands."""
        self.hands = [Hand()]
        self.current_hand_index = 0
    
    @property
    def current_hand(self) -> Hand:
        """Get current active hand."""
        return self.hands[self.current_hand_index]
    
    def add_card(self, card: Card) -> None:
        """Add card to current hand."""
        self.current_hand.add_card(card)
    
    def get_valid_actions(self, dealer_upcard: Card) -> List[Action]:
        """Get list of valid actions for current hand."""
        hand = self.current_hand
        actions = [Action.HIT, Action.STAND]
        
        # Double down allowed on initial two cards
        if hand.can_double() and self.bankroll >= hand.bets[BetType.MAIN]:
            actions.append(Action.DOUBLE)
        
        # Split allowed on pairs if under max hands
        if (hand.can_split() and 
            self.rules.can_split(len(self.hands)) and 
            self.bankroll >= hand.bets[BetType.MAIN]):
            actions.append(Action.SPLIT)
        
        # Insurance only against dealer ace
        if (dealer_upcard.is_ace() and 
            len(hand.cards) == 2 and  # Only on initial hand
            self.bankroll >= hand.bets[BetType.MAIN] / 2):
            actions.append(Action.INSURANCE)
        
        return actions
    
    def split(self) -> None:
        """Split current hand into two hands."""
        if not self.current_hand.can_split():
            raise ValueError("Cannot split current hand")
            
        # Create new hand from split
        new_hand = self.current_hand.split()
        
        # Insert new hand after current hand
        self.hands.insert(self.current_hand_index + 1, new_hand)
        
        # Update bankroll
        self.bankroll -= new_hand.bets[BetType.MAIN]
    
    def double_down(self, card: Card) -> None:
        """Double bet and take one card."""
        hand = self.current_hand
        if not hand.can_double():
            raise ValueError("Cannot double down on current hand")
            
        # Double bet and update bankroll
        self.bankroll -= hand.bets[BetType.MAIN]
        hand.bets[BetType.MAIN] *= 2
        
        # Add card and mark as doubled
        hand.add_card(card)
        hand.status = HandStatus.DOUBLE
    
    def take_insurance(self) -> bool:
        """Take insurance bet."""
        hand = self.current_hand
        insurance_amount = hand.bets[BetType.MAIN] / 2
        
        if insurance_amount > self.bankroll:
            return False
            
        hand.bets[BetType.INSURANCE] = insurance_amount
        self.bankroll -= insurance_amount
        return True
    
    def next_hand(self) -> bool:
        """
        Move to next available hand.
        Returns True if there is a next hand, False otherwise.
        """
        self.current_hand_index += 1
        return self.current_hand_index < len(self.hands)
    
    def calculate_rewards(self, dealer: 'Dealer') -> float:
        """Calculate total rewards for all hands."""
        total_reward = 0
        
        for hand in self.hands:
            # Main bet
            main_payout = self.bet_manager.calculate_main_payout(hand, dealer.hand)
            total_reward += main_payout
            
            # Perfect Pairs
            pp_payout = self.bet_manager.calculate_perfect_pairs_payout(hand)
            total_reward += pp_payout
            
            # 21+3
            plus3_payout = self.bet_manager.calculate_21plus3_payout(hand, dealer.get_upcard())
            total_reward += plus3_payout
            
            # Insurance
            ins_payout = self.bet_manager.calculate_insurance_payout(hand, dealer.hand)
            total_reward += ins_payout
        
        self.bankroll += total_reward
        return total_reward
    
    def is_bust(self) -> bool:
        """Check if all hands are bust."""
        return all(hand.status == HandStatus.BUST for hand in self.hands)
    
    def __str__(self) -> str:
        """String representation showing all hands and bankroll."""
        hands_str = "\n".join(f"Hand {i+1}: {hand}" 
                            for i, hand in enumerate(self.hands))
        return f"Bankroll: ${self.bankroll:.2f}\n{hands_str}" 