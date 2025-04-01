from pettingzoo.classic import hanabi_v5
import numpy as np


def is_card_playable(card_vector, fireworks_state):
    """
    Determines if a card is playable based on the current fireworks state.

    Args:
        card_vector (list): 25-bit one-hot vector representing the card (color*5 + rank)
        fireworks_state (dict): Dictionary containing current firework states for each color

    Returns:
        bool: True if the card is playable, False otherwise
    """
    # Constants
    RANKS_PER_COLOR = 5
    COLORS = ["red", "yellow", "green", "white", "blue"]

    # Find the card's color and rank from the one-hot encoding
    indices = np.where(card_vector == 1)[0]
    if len(indices) == 0:
        return False  # or raise an error
    card_index = indices[0]
    card_color = COLORS[card_index // RANKS_PER_COLOR]
    card_rank = (card_index % RANKS_PER_COLOR) + 1  # Add 1 because ranks start at 1

    # Get the current firework value for the card's color
    firework_start_indices = {
        "red": 175,
        "yellow": 180,
        "green": 185,
        "white": 190,
        "blue": 195,
    }

    # Get the current height of the firework for this color
    firework_vector = fireworks_state[
        firework_start_indices[card_color] : firework_start_indices[card_color] + 5
    ]
    current_firework_height = sum(firework_vector)  # Height is number of 1s in vector

    # Card is playable if its rank is exactly one more than current firework height
    return card_rank == current_firework_height + 1


def first_move_agent(observation, action_space):
    action_mask = observation["action_mask"]
    observation = observation["observation"]
    # Did i receive a hint? If so, play the card
    number_hint_available = any(observation[274:279])
    if number_hint_available:
        playable_card_index = next(
            (i for i, x in enumerate(observation[274:279]) if x == 1), None
        )
        return 5 + playable_card_index

    # Can i give a hint? If so, give that hint
    # do this for each card in my hand
    for i in range(5):
        test = is_card_playable(observation[i * 25 : (i + 1) * 25], observation)
        print(f"Card {i} is playable: {test}")

    # Discard the rightmost card

    """Agent that always selects the first legal move available."""

    # Find the first legal action (where mask value is 1)
    for action, is_legal in enumerate(action_mask):
        if is_legal:
            print(f"Action {action} is legal")
            return action
    return None


# Create and initialize the environment
env = hanabi_v5.env(
    colors=5,
    ranks=5,
    players=2,
    hand_size=5,
    max_information_tokens=8,
    max_life_tokens=3,
    observation_type="minimal",
    render_mode="human",
)
env.reset(seed=42)

# Main game loop
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # Use our first_move_agent to select an action
        action = first_move_agent(observation, env.action_space(agent))

    env.step(action)

env.close()
