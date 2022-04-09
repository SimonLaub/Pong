import gym
import random

import matplotlib.pyplot as plt
from matplotlib import animation

def save_frames_as_gif(frames, filename=None):
    """
    Save a list of frames as a gif
    """
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    if filename:
        anim.save(filename, dpi=72, writer='imagemagick')

# Frame list collector
frames = []
STEPS = 300

# code for the two only actions in Pong
UP_ACTION = 2
DOWN_ACTION = 3

# initializing our environment
env = gym.make("Pong-v0")

# beginning of an episode
observation = env.reset()

# main loop
for i in range(STEPS):
    # choose random action
    action = random.randint(UP_ACTION, DOWN_ACTION)

    #run one step
    observation, reward, done, info = env.step(action)
    frames.append(observation) # collecting observation

    # if episode is over, reset to beginning
    if done:
        observation = env.reset()
        frames.append(observation) # collecting observation
        
# Save the run
save_frames_as_gif(frames, filename='pong-random-300-steps.gif')

