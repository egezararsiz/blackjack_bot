from typing import NamedTuple
from ..core.hand import Hand
from ..core.enums import HandStatus

class HandState(NamedTuple):
    """State representation of a hand for RL agent."""
    value: int
    is_soft: bool
    num_cards: int
    status: HandStatus
    split_from_aces: bool

def get_hand_state(hand: Hand) -> HandState:
    """Convert hand to RL state representation."""
    return HandState(
        value=hand.get_value(),
        is_soft=hand.is_soft(),
        num_cards=len(hand.cards),
        status=hand.status,
        split_from_aces=hand.split_from_aces
    ) 