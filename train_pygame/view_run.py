from model import Environment, load_ai_run

env = Environment()
# ans = load_ai_run(90, "test")
# ans = load_ai_run(80, "closer_small_reward")
ans = load_ai_run(70, "20250712first")
env.play(True, ans["actions"], ans["foods"])

# Test = initial setup, reward +1 for food, otw, reward -.01

# Closer small reward = reward +1 for food, if closer to something than last time reward +.01, otw -.01

# # Assume model is your DQN (a torch.nn.Module)
# torch.save(model.state_dict(), "dqn_weights.pth")

# # Save the model
# model = DQN()  # You need to reinstantiate the same model architecture
# model.load_state_dict(torch.load("dqn_weights.pth"))
# model.eval()  # Important: set to evaluation mode for inference
