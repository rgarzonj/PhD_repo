import util
import numpy as np
import gym
import copy
import random
import time
from QValueFunction import QValueFunction
from skdata.tests.test_caltech import counts_101
from joblib import Parallel, delayed
from util import saveContents
from util import loadContents

MAX_GAME_STEPS = 8000
MAX_SIM_STEPS = 4000
NUM_ROLLOUTS = 100
EPSILON = 0.1
LEARNING_RATE = 0.05

# Number of episodes that runs PlayGame
MAX_EPISODES = 5
# This is not used
#LAMBDA = 100

P_MIN = -1.2;
P_MAX = 0.5;
V_MIN = -0.07;
V_MAX = 0.07;
P_DIF = P_MAX - P_MIN
V_DIF = V_MAX-V_MIN
print V_DIF
print P_DIF

#Simulate the game starting from 'start_state'    

def SimulateGame(env,start_state,qFunction,nSimulation):
    #print 'Starting simulation'
    #print start_state
    #def SimulateGame(start_state):
    #env = gym.make('MountainCar-v0')
#    qFunction.getWeights()
    observedUtility = 0
    currentState = start_state
    currentState = [(currentState[0]-P_MIN)/(P_DIF),(currentState[1]-V_MIN)/(V_DIF)]
    possibleActions = [0,1,2]
    for u in range(MAX_SIM_STEPS):
        if (util.flipCoin(EPSILON)):
            #print 'Choosing random action'
            nextAction = random.choice(possibleActions)
        else:
            #print 'Choosing best action value action'
            qValues = []
            for a in possibleActions:
#                print 'Evaluating action %d' %a
                qValue = qFunction.computeQFunction (a,currentState)
                qValues.append(qValue)
            nextAction = possibleActions[qValues.index(max(qValues))]
        #Execute nextAction
        #print 'Simulation executing action %d' %a
#        env.render()
#        time.sleep(0.01)
        currentState, reward, done, info = env.step(nextAction)
        currentState = [(currentState[0]-P_MIN)/(P_DIF),(currentState[1]-V_MIN)/(V_DIF)]
        observedUtility += reward
        if (done):
            print 'Simulation reached terminal state !!!!!!!!!!!!'
            break
    # Return action and observed utility
    qValues = []
    for a in possibleActions:
        qValue = qFunction.computeQFunction (a,start_state)
        qValues.append(qValue)
    m = max(qValues)
    bestAction = possibleActions[qValues.index(m)]
    #Update parameters w of Q(s,a)
    qFunction.updateWeights(m, observedUtility, bestAction, start_state, LEARNING_RATE)

    print 'Simulation returning %d with action-value %f' %(bestAction , m)
    return bestAction, m

def PlayGame():
    #Initialize game state to fixed starting state
    envir = gym.make('MountainCar-v0')    
    nextState = envir.reset()
    totalReward = 0
    #for t = 1 . . . T do
    for t in range(MAX_GAME_STEPS):
        rewards = np.zeros(envir.action_space.n)
        counting = np.zeros(envir.action_space.n)
#         #Run N simulated games for i = 1 . . . N do    
        qFunction = QValueFunction(envir.action_space.n,nextState.shape[0])
        saveContents(envir,'frozenEnvironment.pckl')
        for i in range(NUM_ROLLOUTS):
            frozenEnv = loadContents('frozenEnvironment.pckl')
            #frozenEnv = copy.copy(envir)
            print 'Starting simulation %d' %i
            a_i,r_i = SimulateGame(frozenEnv,nextState,qFunction,i)
            #print a_i
            #print r_i
            # This is an improvement: computing only the average rewards during the latest
            # rollouts when the weights of the Q function have already converged
            if (i>(NUM_ROLLOUTS*4/5)):
                rewards[a_i] += r_i
                counting[a_i] = counting[a_i]+1
                #print rewards
                #print counting        
        #For every action    
        #Compute average observed utility for each action            
        avg_rewards = rewards/counting
        print avg_rewards
        #Execute selected action in game       
#        selectedAction = avg_rewards.index(max(avg_rewards))
        avg_rewards[np.isnan(avg_rewards)] = 0       
        selectedAction = avg_rewards.argmax()        
        print ('Playing next step %d with action %d +++++++++++++++++++++++++++++++++++++++++++++') %(t,selectedAction)
        #envir.render()
        nextState,reward,done,info = envir.step(selectedAction)
        totalReward+=reward
        if (done):
            print 'Game reached terminal state in %d steps with reward %f' %(t,totalReward)
            break
    print 'Game reached maximum number of steps %f' %totalReward
    #envir.close()
    return (totalReward)
### END OF SCRIPT
#PlayGame()

all_rewards = 0
for episode in range (MAX_EPISODES):
    print '---------------- Starting episode %d ----------------' %episode
    rew = PlayGame()
    all_rewards += rew

avg_rew = all_rewards/float(MAX_EPISODES)
print 'Average reward is %f ' %avg_rew
    
#envir = gym.make('MountainCar-v0')
#initialState = envir.reset()
#action,reward = SimulateGame(envir,initialState)
#print action
#print reward



            