import pytest
import numpy as np
from src.environment.blackjack_env import BlackjackEnv
from src.environment.card import Card
from src.environment.enums import Rank, Suit, Action

@pytest.fixture
def env():
    """Create a test environment."""
    return BlackjackEnv(num_players=1)

def test_env_initialization(env):
    """Test environment initialization."""
    assert env.num_players == 1
    assert env.decks == 8
    assert env.initial_bankroll == 10000
    assert len(env.shoe) == 8 * 52
    assert len(env.seen_cards) == 0

def test_deal_card(env):
    """Test dealing cards."""
    initial_shoe_size = len(env.shoe)
    card = env.deal_card()
    assert isinstance(card, Card)
    assert len(env.shoe) == initial_shoe_size - 1
    assert len(env.seen_cards) == 1

def test_calculate_hand_value():
    """Test hand value calculation."""
    env = BlackjackEnv()
    
    # Test regular cards
    hand = [
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.FIVE, Suit.CLUBS)
    ]
    assert env.calculate_hand_value(hand) == 15
    
    # Test ace handling
    hand = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.KING, Suit.CLUBS)
    ]
    assert env.calculate_hand_value(hand) == 21
    
    # Test multiple aces
    hand = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.ACE, Suit.CLUBS),
        Card(Rank.NINE, Suit.DIAMONDS)
    ]
    assert env.calculate_hand_value(hand) == 21

def test_is_blackjack():
    """Test blackjack detection."""
    env = BlackjackEnv()
    
    # Test natural blackjack
    hand = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.KING, Suit.CLUBS)
    ]
    assert env.is_blackjack(hand) is True
    
    # Test non-blackjack 21
    hand = [
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.FIVE, Suit.CLUBS),
        Card(Rank.SIX, Suit.DIAMONDS)
    ]
    assert env.is_blackjack(hand) is False

def test_can_split():
    """Test split possibility detection."""
    env = BlackjackEnv()
    
    # Test splittable hand
    hand = [
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.KING, Suit.CLUBS)
    ]
    assert env.can_split(hand) is True
    
    # Test non-splittable hand
    hand = [
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.CLUBS)
    ]
    assert env.can_split(hand) is False

def test_perfect_pairs_payout():
    """Test perfect pairs payouts."""
    env = BlackjackEnv()
    
    # Test perfect pair
    hand = [
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS)
    ]
    assert env.calculate_perfect_pairs_payout(hand) == 25
    
    # Test colored pair
    hand = [
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.KING, Suit.DIAMONDS)
    ]
    assert env.calculate_perfect_pairs_payout(hand) == 12
    
    # Test mixed pair
    hand = [
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.KING, Suit.CLUBS)
    ]
    assert env.calculate_perfect_pairs_payout(hand) == 6
    
    # Test non-pair
    hand = [
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.CLUBS)
    ]
    assert env.calculate_perfect_pairs_payout(hand) == 0

def test_reset(env):
    """Test environment reset."""
    state = env.reset()
    
    assert len(env.player_hands[0]) == 2
    assert len(env.dealer_cards) == 2
    assert isinstance(state, np.ndarray)
    assert env.bankroll == env.initial_bankroll

def test_step(env):
    """Test environment step."""
    env.reset()
    
    # Test hit action
    state, reward, done, info = env.step(Action.HIT.value)
    assert len(env.player_hands[0]) == 3
    assert isinstance(state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    # Test stand action
    state, reward, done, info = env.step(Action.STAND.value)
    assert done is True  # Game should end after stand

def test_bankroll_updates(env):
    """Test bankroll updates."""
    initial_bankroll = env.bankroll
    env.reset()
    
    # Simulate a win
    env.wins += 1
    state, reward, done, info = env.step(Action.STAND.value)
    assert env.bankroll > initial_bankroll
    
    # Reset and simulate a loss
    env.reset()
    env.losses += 1
    state, reward, done, info = env.step(Action.STAND.value)
    assert env.bankroll < initial_bankroll 