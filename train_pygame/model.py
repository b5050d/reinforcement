"""
Easy Level Simulation to play and train AI on
"""

# Handle Imports
import pygame
import sys
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import pickle


# Set up the Resources needed
currdir = os.path.dirname(__file__)
sprite_path = os.path.join(currdir,"sprite.png")

ARENA_SIZE = 300
WIDTH, HEIGHT = ARENA_SIZE, ARENA_SIZE
GREEN = (0, 200, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PLAYER_RADIUS = 5
PLAYER_POS = [WIDTH // 2, HEIGHT // 2]
PLAYER_SPEED = 1
FPS = 120
clock = pygame.time.Clock()

def normalize_delta(span, delta):
    ans = (delta / (2*span)) + (.5)
    if ans < 0:
        return 0
    elif ans > 1:
        return 1
    return ans

def find_euclidean_distance(point_a, point_b):
    """
    Find the euclidean distance between 2 points
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    return np.linalg.norm(point_a - point_b)

def find_x_y_delta(player_pos, food_pos):
    """
    Find the X and Y differences between 2 points
    """
    dx = food_pos[0] - player_pos[0]
    dy = food_pos[1] - player_pos[1]
    return dx, dy

def save_ai_run(foods, actions, run_iter, run_tag):
    """
    Save an AI run to the archive
    """
    assert type(foods) is dict
    assert type(actions) is list
    assert type(actions[0]) is int

    run = {}
    run["foods"] = foods
    run["actions"] = actions

    run_iter = f"{run_iter:04d}"

    filepath = os.path.join(os.path.dirname(__file__), "stored", f"{run_tag}_{run_iter}.pkl")

    with open(filepath, "wb") as f:
        pickle.dump(run, f)
    assert os.path.exists(filepath)

def save_ai_model(model, model_tag, model_num):
    """
    Save the AI model for future re-use
    """
    filepath = os.path.join(os.path.dirname(__file__), "stored", f"{model_tag}_{model_num:04d}.pth")
    torch.save(model.state_dict(), filepath)

def load_ai_model(model_class, filepath):
    """
    Load an existing AI model into evaluation mode to see how it does
    """
    assert os.path.exists(filepath)
    model = model_class
    model.load_state_dict(torch.load(filepath))
    model.eval

def load_ai_run(run_iter, run_tag):
    """
    Load an AI run to see how it has done
    """
    run_iter = f"{run_iter:04d}"
    filepath = os.path.join(os.path.dirname(__file__), "stored", f"{run_tag}_{run_iter}.pkl")
    with open(filepath, "rb") as f:
        ans = pickle.load(f)
    return ans

def get_random_foods(width, height, num = 10):
    """
    Get a random arrangement of foods about the arena
    """
    foods = {}
    for i in range(num):
        stop_condition = False
        while not stop_condition:
            x = np.random.randint(20, width- 20)
            y = np.random.randint(20, height - 20)
            rand_pos = (x, y)
            if rand_pos not in foods:
                stop_condition = True
                foods[rand_pos] = 1
    return foods

def encode_foods(player_x, player_y, foods):
    """
    Encode the foods dictionary into an observation space
    digestible by the model
    """

    diffs_active = []
    keys_active = []

    diffs_inactive = []
    keys_inactive = []
    for fpos in foods.keys():
        if foods[fpos] ==  1:
            diffs_active.append(find_euclidean_distance([player_x, player_y], fpos))
            keys_active.append(fpos)
        else:
            diffs_inactive.append(find_euclidean_distance([player_x, player_y], fpos))
            keys_inactive.append(fpos)
    
    sorted_pairs_active = sorted(zip(diffs_active, keys_active))
    sorted_pairs_inactive = sorted(zip(diffs_inactive, keys_inactive))

    foods_array = []

    # Add the players position here
    foods_array.append(player_x/WIDTH)
    foods_array.append(player_y/HEIGHT)

    for euc, fpos in sorted_pairs_active:
        dx, dy = find_x_y_delta([player_x, player_y], fpos)
        foods_array.append(normalize_delta(WIDTH, dx))
        foods_array.append(normalize_delta(HEIGHT, dy))
        foods_array.append(foods[fpos])

    for euc, fpos in sorted_pairs_inactive:
        dx, dy = find_x_y_delta([player_x, player_y], fpos)
        foods_array.append(normalize_delta(WIDTH, dx))
        foods_array.append(normalize_delta(HEIGHT, dy))
        foods_array.append(foods[fpos])
    
    # print(foods_array)
    assert max(foods_array) <= 1
    assert min(foods_array) >= 0

    return np.array(foods_array, dtype=np.float32)

class Environment():
    """
    The Environment that runs the game
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the Game
        """
        # Get new foods
        self.foods = get_random_foods(WIDTH, HEIGHT, 10)
        self.player_pos = [int(WIDTH/2), int(HEIGHT/2)]

        self.last_min_euclid = 999999
        return encode_foods(self.player_pos[0], self.player_pos[1], self.foods)

    def step(self, action):
        """
        Take an action in the game, move the player,
        check if the player has reached a food,
        check if the player is done, return next state
        """
        if action == 0:  #up
            self.player_pos[1]-=PLAYER_SPEED
        elif action == 1: # Upper Left
            self.player_pos[0]-=PLAYER_SPEED
            self.player_pos[1]-=PLAYER_SPEED
        elif action == 2: # Left
            self.player_pos[0]-=PLAYER_SPEED
        elif action == 3: # Down Left
            self.player_pos[0]-=PLAYER_SPEED
            self.player_pos[1]+=PLAYER_SPEED
        elif action == 4: # Down
            self.player_pos[1]+=PLAYER_SPEED
        elif action == 5: # Down Right
            self.player_pos[0]+=PLAYER_SPEED
            self.player_pos[1]+=PLAYER_SPEED
        elif action == 6: # Right
            self.player_pos[0]+=PLAYER_SPEED
        elif action == 7: # Up Right
            self.player_pos[0]+=PLAYER_SPEED
            self.player_pos[1]-=PLAYER_SPEED
        else:
            pass

        if self.player_pos[0] >= WIDTH:
            self.player_pos[0] = WIDTH-1
        elif self.player_pos[0] <= 0:
            self.player_pos[0]=0

        if self.player_pos[1] >= HEIGHT:
            self.player_pos[1] = HEIGHT-1
        elif self.player_pos[1] <= 0:
            self.player_pos[1]=0


        reward = 0
        # Check if we are in proximity of the food
        euclids = []
        for f in self.foods:
            if self.foods[f] == 1: # active
                ans = find_euclidean_distance(self.player_pos, f)
                euclids.append(ans)
                if ans <= 5:
                    self.foods[f] = 0
                    reward+=1

        # Check if we are closer or further than last time:
        # Get the min distance
        min_euclid = min(euclids)
        difference = min_euclid - self.last_min_euclid
        if difference < 0:
            reward += -.1 * difference
        if reward == 0:
            reward = -.01
        self.last_min_euclid = min_euclid
            
        # # Encode the Foods
        obs = encode_foods(self.player_pos[0], self.player_pos[1], self.foods)
        
        # Check how many foods remain
        done = True
        for f in self.foods:
            if self.foods[f] == 1:
                done = False
                break
        if done == True:
            reward = 10 # won the game! give a big reward
        return obs, reward, done

    def handle_keypresses(self):
        """
        Handle the Keypresses
        """
        keys = pygame.key.get_pressed()
        key_flags = [keys[pygame.K_w], keys[pygame.K_a], keys[pygame.K_s],  keys[pygame.K_d]]
        value = 0
        for i, bit in enumerate(reversed(key_flags)):
            value |= bit << i

        if value in [8, 13]:
            action = 0
        elif value == 12:
            action = 1
        elif value in [4, 14]:
            action = 2
        elif value == 6:
            action = 3
        elif value in [2, 7]:
            action = 4
        elif value == 3:
            action = 5
        elif value in [1, 11]:
            action = 6
        elif value == 9:
            action = 7
        else:
            action = -1

        return action

    def play(self, ai = False, ai_history = [], ai_foods = {}):
        if ai:
            # Get the player back to the starting point
            self.player_pos = [int(WIDTH/2), int(HEIGHT/2)]

            # Make all the foods active again
            for f in self.foods:
                self.foods[f] = 1
        else:
            self.reset()

        # Render the environment
        pygame.init()

        font = pygame.font.SysFont(None, 24)

        # Setup display
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Arena")

        # Load sprite
        player_image = pygame.image.load(sprite_path).convert_alpha()
        player_rect = player_image.get_rect(center=(WIDTH // 2, HEIGHT // 2))

        first_move = True
        game_timer = 0
        step_counter = 0
        
        stop_condition = False
        while not stop_condition:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Handle action
            if ai:
                action = ai_history[step_counter]
            else:
                action = self.handle_keypresses()

            if first_move:
                if action != -1:
                    first_move = False

            next_state, reward, done = self.step(action)

            # Rendering Steps
            screen.fill(GREEN)

            # Draw Player
            player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))
            screen.blit(player_image, player_rect)

            for f in self.foods:
                if self.foods[f] == 1:
                    pygame.draw.circle(screen, YELLOW, f, 5)

            clock.tick(FPS)

            step_counter+=1
            if not first_move:
                game_timer+=1/FPS

            text_clock = font.render(f"{game_timer:.2f}", True, (255, 255, 255))
            text_rect = text_clock.get_rect(topright=(WIDTH-10, 10))

            screen.blit(text_clock, text_rect)

            # Update display
            pygame.display.flip()

            if ai:
                if step_counter >= len(ai_history):
                    done = True

            if done:
                print("Finished game!")
                stop_condition = True

        print(f"The time taken for you was: {game_timer}")
        print("Game Over!")


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self, x):
        return self.model(x)
    

def training_loop():
    env = Environment()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")

    nnet = DQN()
    nnet.to(device)
    LEARNING_RATE = 1e-3
    optimizer = optim.Adam(nnet.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.MSELoss()

    # Set up the Target Network
    nnet_target = DQN()
    nnet_target.load_state_dict(nnet.state_dict())
    nnet_target.eval()  # no gradients needed


    replay_buffer = deque(maxlen=10000)
    batch_size = 64
    gamma = .99
    epsilon = 1.0
    epsilon_decay = .995
    epsilon_min = .1
    episodes = 500

    total_steps = 0

    EVALUATION_INTERVAL = 10
    TARGET_UPDATE_N = 5 # Update the target network every N episodes

    for ep in range(episodes):
        start_time = time.time()
        print(f"Beginning Episode: {ep}")
        state = env.reset()
        total_reward = 0

        action_history = []
        for t in range(2000):
            if random.random() < epsilon:
                action = random.randint(0, 7)
            else:
                with torch.no_grad():
                    q_vals = nnet(torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device))
                    action = torch.argmax(q_vals).item()

            action_history.append(action)

            next_state, reward, done = env.step(action)
            replay_buffer.append(
                (state, action, reward, next_state, done)
            )
            state = next_state
            total_reward += reward

            # Train
            if (len(replay_buffer) >= batch_size) and (total_steps > 1000):
                batch = random.sample(replay_buffer, batch_size)
                s, a, r, s2, d = zip(*batch)
                s = np.array(s)
                a = np.array(a)
                r = np.array(r)
                s2 = np.array(s2)
                d = np.array(d)

                s = torch.tensor(s, dtype=torch.float32).to(device)
                a = torch.tensor(a).to(device)
                r = torch.tensor(r, dtype=torch.float32).to(device)
                s2 = torch.tensor(s2, dtype=torch.float32).to(device)
                d = torch.tensor(d, dtype=torch.float32).to(device)

                qvals = nnet(s)
                qval = qvals.gather(1, a.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    max_q_s2 = nnet_target(s2).max(1)[0]
                    target = r + gamma * max_q_s2 * (1 - d)

                loss = loss_fn(qval, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_steps+=1

            if done:
                break
                

        if ep%TARGET_UPDATE_N == 0:  # or every N episodes or steps
            nnet_target.load_state_dict(nnet.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {ep}, reward: {total_reward:.2f}, epsilon: {epsilon:.3f}")
        print(f"Episode Run time = {round(time.time() - start_time, 3)}")

        # Save the run every 10
        if ep%EVALUATION_INTERVAL == 0:
            # tag = "closer_bigbig_reward"
            # save_ai_run(env.foods, action_history, ep, tag)
            # save_ai_model(nnet, tag, ep)
            run_tag = "testing"
            evaluation_loop(run_tag, ep, nnet)

            pygame.quit()

def evaluation_loop(run_tag, env, ep, nnet):
    nnet.eval()

    episodes = 1
    reward = 0
    for ep in range(episodes):
        # Reset the env
        env.reset_to_eval()
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            state, reward, done = env.step(action)
            episode_reward += reward

        total_reward += episode_reward

    avg_reward = total_reward / episodes
    print(f"[Eval] Avg reward over {episodes} eval runs: {avg_reward:.2f}")

    # TODO - Save the model
        # tag = "closer_bigbig_reward"
    # save_ai_run(env.foods, action_history, ep, tag)
    # save_ai_model(nnet, tag, ep)
    nnet.train() # Set it back in training mode


def evaluation_loop():
    pass

def play_loop():
    env = Environment()
    env.play()


if __name__ == "__main__":
    play_loop()
    # training_loop()
    # env = Environment()
    # ans = load_ai_run(0, "test")
    # env.play(True, ans["actions"], ans["foods"])




    # def play_human(self, render = False):
    #     """
    #     Play (as a human)
    #     """
    #     self.reset()

    #     # Render the environment so the user can play
    #     pygame.init()

    #     # set up the display clock
    #     font = pygame.font.SysFont(None, 36)

    #     # Setup display
    #     screen = pygame.display.set_mode((WIDTH, HEIGHT))
    #     pygame.display.set_caption("Arena")

    #     # Load sprite
    #     player_image = pygame.image.load(sprite_path).convert_alpha()
    #     player_rect = player_image.get_rect(center=(WIDTH // 2, HEIGHT // 2))

    #     # Game loop
    #     cooldown = 0
    #     first_move = True
    #     timer_clock_val = 0

    #     timer_clock_val
    #     # x = 3.1415926
    #     # print(f"{x:.3f}")
    #     while True:
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 pygame.quit()
    #                 sys.exit()

    #         # Handle keypresses
    #         keys = pygame.key.get_pressed()
    #         d = False
    #         r = 0

    #         if keys[pygame.K_w]:
    #             # PLAYER_POS[1] -= PLAYER_SPEED
    #             o, r, d = self.step(0)
    #         if keys[pygame.K_s]:
    #             # PLAYER_POS[1] += PLAYER_SPEED
    #             o, r, d = self.step(2)
    #         if keys[pygame.K_a]:
    #             # PLAYER_POS[0] -= PLAYER_SPEED
    #             o, r, d = self.step(1)
    #         if keys[pygame.K_d]:
    #             # PLAYER_POS[0] += PLAYER_SPEED
    #             o, r, d = self.step(3)

    #         if first_move:
    #             print("Encountered the first move")
    #             if keys[pygame.K_w] or keys[pygame.K_a] or keys[pygame.K_s] or keys[pygame.K_d]:
    #                 first_move = False
    #                 first_move += 1/FPS
    #         else:
    #             first_move += 1/FPS
            

    #         if keys[pygame.K_x]:
    #             if cooldown == 0:
    #                 print(PLAYER_POS)
    #                 cooldown+=600

    #         # Fill background
    #         screen.fill(GREEN)

    #         # Draw player
    #         # pygame.draw.circle(screen, RED, PLAYER_POS, PLAYER_RADIUS)
    #         player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))
    #         screen.blit(player_image, player_rect)

    #         for f in self.foods:
    #             if self.foods[f] == 1:
    #                 pygame.draw.circle(screen, YELLOW, f, 5)

    #         if d:
    #             print("Finished game!")
    #             break

    #         # Handle the Timer Clock
    #         timer_string = print(f"{timer_clock_val:.3f}")
    #         text_clock = font.render(timer_string, True, (255, 0, 0)) 
    #         text_rect = text_clock.get_rect(topright=(WIDTH-10, 10))
        
    #         screen.blit(text_clock,text_rect)

    #         # Update display
    #         pygame.display.flip()

    #         if cooldown>0:
    #             cooldown=cooldown-1

    #         clock.tick(FPS)

    #     print(f"The time taken for you was: {timer_clock_val}")
    #     print("Game Over!")

    # def render(self, foods, action_history):
    #     """
    #     Render a game played by AI
    #     """
    #     # Get the player back to the starting point
    #     self.player_pos = [int(WIDTH/2), int(HEIGHT/2)]

    #     # Make all the foods active again
    #     for f in self.foods:
    #         self.foods[f] = 1

    #     # Render the environment so the user can play
    #     pygame.init()

    #     # Setup display
    #     screen = pygame.display.set_mode((WIDTH, HEIGHT))
    #     pygame.display.set_caption("Arena")

    #     # Load sprite
    #     player_image = pygame.image.load(sprite_path).convert_alpha()
    #     player_rect = player_image.get_rect(center=(WIDTH // 2, HEIGHT // 2))

    #     # Game loop
    #     cooldown = 0
    #     start_time = time.time()
    #     move_ticker = 0
    #     while True:
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 pygame.quit()
    #                 sys.exit()

    #         # Handle keypresses
    #         # keys = pygame.key.get_pressed()
    #         d = False
    #         r = 0

    #         if move_ticker == len(action_history)-1:
    #             d = True

    #         action = action_history[move_ticker]
    #         self.step(action)
    #         move_ticker+=1
            
    #         # if keys[pygame.K_w]:
    #         #     # PLAYER_POS[1] -= PLAYER_SPEED
    #         #     o, r, d = self.step(0)
    #         # if keys[pygame.K_s]:
    #         #     # PLAYER_POS[1] += PLAYER_SPEED
    #         #     o, r, d = self.step(2)
    #         # if keys[pygame.K_a]:
    #         #     # PLAYER_POS[0] -= PLAYER_SPEED
    #         #     o, r, d = self.step(1)
    #         # if keys[pygame.K_d]:
    #         #     # PLAYER_POS[0] += PLAYER_SPEED
    #         #     o, r, d = self.step(3)

    #         # if keys[pygame.K_x]:
    #         #     if cooldown == 0:
    #         #         print(PLAYER_POS)
    #         #         cooldown+=600

    #         # Fill background
    #         screen.fill(GREEN)

    #         # Draw player
    #         # pygame.draw.circle(screen, RED, PLAYER_POS, PLAYER_RADIUS)
    #         player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))
    #         screen.blit(player_image, player_rect)

    #         for f in self.foods:
    #             if self.foods[f] == 1:
    #                 pygame.draw.circle(screen, YELLOW, f, 5)

    #         if d:
    #             print("Finished game!")
    #             break

    #         # Update display
    #         pygame.display.flip()

    #         if cooldown>0:
    #             cooldown=cooldown-1

    #         clock.tick(FPS)

    #     end_time = time.time()
    #     elapsed = round(end_time - start_time,3)

    #     print(f"The time taken for you was: {elapsed}")
    #     print("Game Over!")

