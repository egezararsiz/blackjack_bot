from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from ..core.deck import Deck
from ..core.hand import Hand
from ..core.card import Card
from ..core.rules import BlackjackRules
from ..core.enums import HandStatus, BetType, Action
from .dealer import Dealer
from .player import Player
from .bet import BetManager, Bet

@dataclass
class Table:
    """
    Coordinates all game components and manages game flow for blackjack.
    Implements Pragmatic Play's rules:
    - 8 decks, shuffle at 50% penetration
    - Dealer stands on all 17s
    - One split allowed
    - No double after split
    - Split aces get one card each
    - Blackjack pays 3:2
    - Insurance pays 2:1
    - Perfect Pairs and 21+3 side bets available
    """
    num_spots: int
    min_bet: float
    max_bet: float
    num_decks: int = field(default=8)
    rules: BlackjackRules = field(default_factory=BlackjackRules)
    
    def __post_init__(self):
        """Initialize table components."""
        self.deck = Deck(self.num_decks)
        self.dealer = Dealer()
        self.bet_manager = BetManager(self.rules, self.min_bet, self.max_bet)
        self.players = [Player() for _ in range(self.num_spots)]
        
        # Game state
        self.current_player_index = 0
        self.round_in_progress = False
        self.insurance_offered = False
        self.all_hands_completed = False
    
    def reset(self) -> None:
        """Reset table for new round."""
        self.dealer.reset()
        for player in self.players:
            player.reset()
            
        self.current_player_index = 0
        self.round_in_progress = False
        self.insurance_offered = False
        self.all_hands_completed = False
        
        # Shuffle if needed based on penetration
        if self.deck.get_penetration() >= 0.5:  # 50% penetration for online play
            self.deck.reset()
    
    def place_bets(self, spot_index: int, main_bet: float,
                   perfect_pairs_bet: float = 0.0,
                   twentyone_plus_three_bet: float = 0.0) -> bool:
        """
        Place bets for a specific spot.
        Returns True if bets are valid and placed successfully.
        """
        if not 0 <= spot_index < self.num_spots:
            return False
            
        player = self.players[spot_index]
        
        # Validate bet amounts
        if not self.bet_manager.validate_bet(main_bet, BetType.MAIN):
            return False
        if not self.bet_manager.validate_bet(perfect_pairs_bet, BetType.PERFECT_PAIRS):
            return False
        if not self.bet_manager.validate_bet(twentyone_plus_three_bet, BetType.TWENTYONE_PLUS_THREE):
            return False
            
        # Check player has enough funds
        total_bet = main_bet + perfect_pairs_bet + twentyone_plus_three_bet
        if player.bankroll < total_bet:
            return False
            
        # Create bet and hand
        bet = Bet(main=main_bet, 
                 perfect_pairs=perfect_pairs_bet,
                 twentyone_plus_three=twentyone_plus_three_bet)
        player.place_bet(bet)
        player.bankroll -= total_bet
        return True
    
    def start_round(self) -> bool:
        """
        Begin new round by dealing initial cards.
        Returns True if round started successfully.
        """
        if self.round_in_progress:
            return False
            
        # Check all active spots have main bets
        for player in self.players:
            if player.has_active_bet() and not player.get_bet().main > 0:
                return False
                
        # Deal initial cards
        for _ in range(2):
            for player in self.players:
                hand = player.get_active_hand()
                if hand and hand.status == HandStatus.ACTIVE:
                    card = self.deck.deal()
                    if not card:
                        return False
                    hand.add_card(card)
                    
            card = self.deck.deal()
            if not card:
                return False
            self.dealer.add_card(card)
                
        self.round_in_progress = True
        
        # Evaluate initial state
        dealer_upcard = self.dealer.get_upcard()
        if dealer_upcard and dealer_upcard.is_ace():
            self.insurance_offered = True
            
        # Evaluate side bets and naturals
        for player in self.players:
            hand = player.get_active_hand()
            bet = player.get_bet()
            if hand and bet:
                # Perfect Pairs
                if bet.perfect_pairs > 0:
                    pair_type = hand.get_perfect_pairs_type()
                    payout = self.bet_manager.calculate_side_bet_payouts(
                        bet, pair_type, PokerHand.NONE)
                    player.bankroll += payout
                    
                # 21+3
                if bet.twentyone_plus_three > 0 and dealer_upcard:
                    poker_hand = hand.evaluate_21plus3(dealer_upcard)
                    payout = self.bet_manager.calculate_side_bet_payouts(
                        bet, PairType.NONE, poker_hand)
                    player.bankroll += payout
                    
        return True
    
    def get_current_player(self) -> Optional[Player]:
        """Get current active player or None if no player is active."""
        if not self.round_in_progress or self.all_hands_completed:
            return None
        return self.players[self.current_player_index]
    
    def get_valid_actions(self) -> Set[Action]:
        """Get valid actions for current player based on dealer's upcard."""
        actions = set()
        player = self.get_current_player()
        if not player:
            return actions
            
        hand = player.get_active_hand()
        if not hand or hand.status != HandStatus.ACTIVE:
            return actions
            
        actions.add(Action.STAND)
        
        # Basic actions
        if hand.get_value() <= 21:
            actions.add(Action.HIT)
            
        # Double down
        if (len(hand.cards) == 2 and
            hand.status == HandStatus.ACTIVE and  # Not split
            player.bankroll >= player.get_bet().main):
            actions.add(Action.DOUBLE)
            
        # Split
        if (hand.is_pair() and
            hand.status == HandStatus.ACTIVE and  # Not already split
            player.can_split() and  # Check rules and count
            player.bankroll >= player.get_bet().main):
            actions.add(Action.SPLIT)
            
        # Insurance
        if (self.insurance_offered and
            not player.has_insurance and
            player.bankroll >= player.get_bet().main * 0.5):
            actions.add(Action.INSURANCE)
            
        return actions
    
    def execute_action(self, action: Action) -> Tuple[bool, str]:
        """
        Process player action and return (success, message).
        Enforces Pragmatic Play's rules on splits and doubles.
        """
        player = self.get_current_player()
        if not player:
            return False, "No active player"
            
        hand = player.get_active_hand()
        if not hand or hand.status != HandStatus.ACTIVE:
            return False, "No active hand"
            
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            return False, f"Invalid action: {action.name}"
            
        if action == Action.STAND:
            hand.stand()
            self._next_player_or_hand()
            return True, "Stand"
            
        elif action == Action.HIT:
            card = self.deck.deal()
            if not card:
                return False, "No cards remaining"
            hand.add_card(card)
            
            if hand.status in [HandStatus.BUST, HandStatus.STAND]:
                self._next_player_or_hand()
            return True, f"Hit: {card}"
            
        elif action == Action.DOUBLE:
            card = self.deck.deal()
            if not card:
                return False, "No cards remaining"
                
            # Take double down bet
            bet = player.get_bet()
            player.bankroll -= bet.main
            bet.double_bet(BetType.MAIN)
            
            # Add card and update status
            hand.double()
            hand.add_card(card)
            self._next_player_or_hand()
            return True, f"Double: {card}"
            
        elif action == Action.SPLIT:
            # Create new hand and bet
            new_hand = hand.split()
            new_bet = Bet(main=player.get_bet().main)
            player.add_split_hand(new_hand, new_bet)
            player.bankroll -= new_bet.main
            
            # Deal one card to each hand
            for h in [hand, new_hand]:
                card = self.deck.deal()
                if not card:
                    return False, "No cards remaining"
                h.add_card(card)
                
            return True, "Split hands"
            
        elif action == Action.INSURANCE:
            bet = player.get_bet()
            insurance_bet = bet.main * 0.5
            player.bankroll -= insurance_bet
            bet.place_bet(BetType.INSURANCE, insurance_bet)
            player.has_insurance = True
            return True, "Insurance taken"
            
        return False, "Unknown action"
    
    def _next_player_or_hand(self) -> None:
        """Advance to next player or hand if current player is done."""
        while True:
            player = self.get_current_player()
            if not player:
                break
                
            # Try next hand for current player
            if player.next_hand():
                break
                
            # Move to next player
            self.current_player_index += 1
            if self.current_player_index >= self.num_spots:
                self.all_hands_completed = True
                break
    
    def play_dealer_hand(self) -> None:
        """
        Play out dealer's hand according to rules.
        Dealer stands on all 17s.
        """
        if not self.round_in_progress or not self.all_hands_completed:
            return
            
        # Dealer must draw to 17
        while self.dealer.should_hit():
            card = self.deck.deal()
            if not card:
                return
            self.dealer.add_card(card)
    
    def settle_bets(self) -> Dict[int, float]:
        """
        Calculate and return rewards for each player.
        Implements Pragmatic Play's payout rules.
        """
        if not self.round_in_progress or not self.all_hands_completed:
            return {}
            
        rewards = {}
        dealer_bj = self.dealer.has_blackjack()
        dealer_value = self.dealer.get_value()
        dealer_bust = dealer_value > 21
        
        for i, player in enumerate(self.players):
            total_reward = 0
            
            # Handle insurance first
            bet = player.get_bet()
            if bet and bet.insurance > 0:
                total_reward += self.bet_manager.calculate_insurance_payout(
                    bet, dealer_bj)
                
            # Process each hand
            for hand in player.hands:
                if not hand:
                    continue
                    
                bet = player.get_bet_for_hand(hand)
                if not bet:
                    continue
                    
                # Calculate main bet payout
                total_reward += self.bet_manager.calculate_payout(
                    bet, hand.status, dealer_bj)
                        
            player.bankroll += total_reward
            rewards[i] = total_reward
            
        self.round_in_progress = False
        return rewards
    
    def get_state(self) -> Dict:
        """Get current game state."""
        return {
            'round_in_progress': self.round_in_progress,
            'dealer': str(self.dealer),
            'current_player': self.current_player_index,
            'players': [str(p) for p in self.players],
            'deck_penetration': self.deck.get_penetration(),
            'insurance_offered': self.insurance_offered,
            'all_hands_completed': self.all_hands_completed
        }
    
    def __str__(self) -> str:
        """String representation of table state."""
        lines = [
            f"Dealer: {self.dealer}",
            f"Deck penetration: {self.deck.get_penetration():.2%}",
            "Players:"
        ]
        for i, player in enumerate(self.players):
            lines.append(f"  {i}: {player}")
        return "\n".join(lines) 