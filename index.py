from pettingzoo.classic import hanabi_v5


def first_move_agent(observation, action_space):
    """Agent that always selects the first legal move available."""
    action_mask = observation["action_mask"]
    # Find the first legal action (where mask value is 1)
    for action, is_legal in enumerate(action_mask):
        if is_legal:
            print(f"Action {action} is legal")
            return action
    return None


# Create and initialize the environment
env = hanabi_v5.env(render_mode="human")
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
