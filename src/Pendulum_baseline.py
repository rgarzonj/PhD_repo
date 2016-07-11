import gym

MAX_EPISODES = 100
MAX_GAME_STEPS = 200

def PlayGame():
    #Initialize game state to fixed starting state
    envir = gym.make('Pendulum-v0')    
    nextState = envir.reset()
    totalReward = 0
    for t in range(MAX_GAME_STEPS):
        nextState,reward,done,info = envir.step(envir.action_space.sample())        
        totalReward+=reward
    envir.close()
    return (totalReward)

all_rewards = 0
for episode in range (MAX_EPISODES):
    print '---------------- Starting episode %d ----------------' %episode
    rew = PlayGame()
    all_rewards += rew

avg_rew = all_rewards/float(MAX_EPISODES)
print 'Average reward is %f ' %avg_rew