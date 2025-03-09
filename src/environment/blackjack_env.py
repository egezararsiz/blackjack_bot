import numpy as np
import gym
from gym import spaces
from .card import Card
from .enums import Rank, Suit, Action

class BlackjackEnv(gym.Env):
    """
    A Blackjack environment implementing the OpenAI Gym interface.
    """
    def __init__(self, num_players=1, num_decks=8, initial_bankroll=10000):
        super().__init__()
        
        self.num_players = num_players
        self.decks = num_decks
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        
        # Initialize metrics
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        
        # Action space: hit, stand, double, split
        self.action_space = spaces.Discrete(4)
        
        # Observation space: player cards, dealer up card, count, bankroll
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([31, 11, 52, float('inf')], dtype=np.float32)
        )
        
        self.reset()
    
    def reset(self):
        """Reset the environment to start a new game."""
        self.shoe = self._create_shoe()
        self.seen_cards = []
        self.player_hands = [[] for _ in range(self.num_players)]
        self.dealer_cards = []
        
        # Deal initial cards
        for _ in range(2):
            for hand in self.player_hands:
                hand.append(self.deal_card())
            self.dealer_cards.append(self.deal_card())
        
        return self._get_observation()
    
    def _create_shoe(self):
        """Create a new shoe with the specified number of decks."""
        shoe = []
        for _ in range(self.decks):
            for suit in Suit:
                for rank in Rank:
                    shoe.append(Card(rank, suit))
        np.random.shuffle(shoe)
        return shoe
    
    def deal_card(self):
        """Deal a single card from the shoe."""
        if not self.shoe:
            self.shoe = self._create_shoe()
        card = self.shoe.pop()
        self.seen_cards.append(card)
        return card
    
    def calculate_hand_value(self, hand):
        """Calculate the value of a hand, handling aces optimally."""
        value = 0
        aces = 0
        
        for card in hand:
            if card.rank == Rank.ACE:
                aces += 1
            else:
                value += min(10, card.rank.value)
        
        for _ in range(aces):
            if value + 11 <= 21:
                value += 11
            else:
                value += 1
        
        return value
    
    def is_blackjack(self, hand):
        """Check if a hand is a natural blackjack."""
        return len(hand) == 2 and self.calculate_hand_value(hand) == 21
    
    def _get_observation(self):
        """Get the current state observation."""
        player_total = self.calculate_hand_value(self.player_hands[0])
        dealer_up_card = self.dealer_cards[0].rank.value
        running_count = self._calculate_running_count()
        
        return np.array([
            player_total,
            dealer_up_card,
            running_count,
            self.bankroll
        ], dtype=np.float32)
    
    def _calculate_running_count(self):
        """Calculate the running count based on seen cards."""
        count = 0
        for card in self.seen_cards:
            if card.rank.value >= 2 and card.rank.value <= 6:
                count += 1
            elif card.rank.value >= 10 or card.rank == Rank.ACE:
                count -= 1
        return count
    
    def step(self, action):
        """
        Take an action in the environment.
        Returns: observation, reward, done, info
        """
        done = False
        reward = 0
        info = {}
        
        if action == Action.HIT.value:
            self.player_hands[0].append(self.deal_card())
            player_value = self.calculate_hand_value(self.player_hands[0])
            
            if player_value > 21:
                done = True
                reward = -1
                self.losses += 1
                self.bankroll -= 1
        
        elif action == Action.STAND.value:
            done = True
            player_value = self.calculate_hand_value(self.player_hands[0])
            dealer_value = self._play_dealer()
            
            if dealer_value > 21 or player_value > dealer_value:
                reward = 1
                self.wins += 1
                self.bankroll += 1
            elif dealer_value > player_value:
                reward = -1
                self.losses += 1
                self.bankroll -= 1
            else:
                reward = 0
                self.pushes += 1
        
        return self._get_observation(), reward, done, info
    
    def _play_dealer(self):
        """Play out the dealer's hand according to fixed rules."""
        while self.calculate_hand_value(self.dealer_cards) < 17:
            self.dealer_cards.append(self.deal_card())
        return self.calculate_hand_value(self.dealer_cards)
    
    def render(self, mode='human'):
        """Render the current state of the game."""
        if mode == 'human':
            print(f"\nPlayer's hand: {self.player_hands[0]}")
            print(f"Player's total: {self.calculate_hand_value(self.player_hands[0])}")
            print(f"Dealer's up card: {self.dealer_cards[0]}")
            print(f"Bankroll: ${self.bankroll}") 