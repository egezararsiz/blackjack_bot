from typing import NamedTuple, Dict
from ..core.enums import BetType
from ..game.bet import Bet, BetManager

class BetState(NamedTuple):
    """State representation of bets for RL agent."""
    main: float
    perfect_pairs: float
    twentyone_plus_three: float
    insurance: float

class BetLimits(NamedTuple):
    """Normalized bet limits for RL agent."""
    min_bet: float
    max_bet: float

def get_bet_state(bet: Bet) -> BetState:
    """Convert bet to RL state representation."""
    return BetState(
        main=bet.main,
        perfect_pairs=bet.perfect_pairs,
        twentyone_plus_three=bet.twentyone_plus_three,
        insurance=bet.insurance
    )

def get_bet_limits(manager: BetManager) -> BetLimits:
    """Get normalized bet limits for RL agent."""
    return BetLimits(
        min_bet=manager.min_bet / manager.max_bet,
        max_bet=1.0
    )

def calculate_reward(bet: Bet, win_multipliers: Dict[str, float]) -> float:
    """Calculate reward for RL agent from bet outcomes."""
    reward = 0.0
    
    # Main bet
    if 'main' in win_multipliers:
        reward += bet.main * win_multipliers['main']
        
    # Perfect pairs
    if 'perfect_pairs' in win_multipliers:
        reward += bet.perfect_pairs * win_multipliers['perfect_pairs']
        
    # 21+3
    if 'twentyone_plus_three' in win_multipliers:
        reward += bet.twentyone_plus_three * win_multipliers['twentyone_plus_three']
        
    # Insurance
    if 'insurance' in win_multipliers:
        reward += bet.insurance * win_multipliers['insurance']
        
    return reward 