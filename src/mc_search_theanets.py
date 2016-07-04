import util
import numpy as np
import gym
import copy
import random
import time
from QNetwork_theanets import QValueNetwork
#from skdata.tests.test_caltech import counts_101
#from joblib import Parallel, delayed

MAX_GAME_STEPS = 400
MAX_SIM_STEPS = 6000
NUM_ROLLOUTS = 5
EPSILON = 0.1
MAX_EPISODES = 5


P_MIN = -1.2;
P_MAX = 0.5;
V_MIN = -0.07;
V_MAX = 0.07;
t = 0

#Simulate the game starting from 'start_state'    
def SimulateGame(env,start_state,qNetwork):
    #print 'Starting simulation'
    #print start_state
    #def SimulateGame(start_state):
    #env = gym.make('MountainCar-v0')
    qNetwork.resetTrainSet()
    global t
    observedUtility = 0
    currentState = start_state
    currentState = [(currentState[0]-P_MIN)/(P_MAX-P_MIN),(currentState[1]-V_MIN)/(V_MAX-V_MIN)]
    possibleActions = [0,1,2]
  
    for u in range(MAX_SIM_STEPS):
        #eps = EPSILON
        #eps = EPSILON/(t/20+1)
        if (util.flipCoin(EPSILON)):
            #print 'Choosing random action'
            nextAction = random.choice(possibleActions)
        else:
            #print 'Choosing best action value action'
            qValues = []
            for a in possibleActions:
#                print 'Evaluating action %d' %a
                qValue = qNetwork.computeQFunction (a,currentState)
                qValues.append(qValue)
            #print qValues
            #if (qValues[0] == qValues[1] and qValues[1]==qValues[2]):
            #    nextAction = 1
            #else:
            nextAction = possibleActions[qValues.index(max(qValues))]
        #Execute nextAction
        #print 'Simulation executing action %d' %nextAction
        #env.render()
#        time.sleep(0.01)
        currentState, reward, done, info = env.step(nextAction)
        currentState = [(currentState[0]-P_MIN)/(P_MAX-P_MIN),(currentState[1]-V_MIN)/(V_MAX-V_MIN)]
        observedUtility += reward
        if (done):
            print 'Simulation reached terminal state after %d steps !!!!!!!!!!!!' %u
            break
    # Return action and observed utility
    
    qValues = []
    for a in possibleActions:
        qValue = qNetwork.computeQFunction (a,start_state)
        qValues.append(qValue)
    bestAction = possibleActions[qValues.index(max(qValues))]    
    #Update parameters w of Q(s,a)
    #qNetwork.resetTrainSet()
    qNetwork.collectSample(observedUtility, bestAction, start_state)
    qNetwork.trainNetwork()
#    qNetwork.trainNetwork()
    m = max(qValues)
    #print 'Simulation returning %d with action-value %f' %(bestAction , m)
    return bestAction, m

def PlayGame():
    global t
    #Initialize game state to fixed starting state
    envir = gym.make('MountainCar-v0')    
    nextState = envir.reset()
    totalReward = 0
    #for t = 1 . . . T do
#    qNetwork = QValueNetwork(envir.action_space.n,nextState.shape[0])
    for t in range(MAX_GAME_STEPS):
        rewards = np.zeros(envir.action_space.n)
        counting = np.zeros(envir.action_space.n)
#         #Run N simulated games for i = 1 . . . N do    
#        qNetwork = QValueNetwork(envir.action_space.n,nextState.shape[0])
        for i in range(NUM_ROLLOUTS):
            frozenEnv = copy.copy(envir)
            print 'Starting simulation %d' %i
            a_i,r_i = SimulateGame(frozenEnv,nextState,qNetwork)
            #print a_i
            #print r_i
            rewards[a_i] += r_i
            counting[a_i] = counting[a_i]+1
            print rewards
            print counting
        #qNetwork.trainNetwork()
        #qNetwork.resetTrainSet()
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
        #qNetwork.resetNetwork()
        totalReward+=reward
        if (done):
            print 'Game reached terminal state in %d with reward' %t %totalReward
            break
    print 'Game reached maximum number of steps %f' %totalReward
    envir.close()
    return (totalReward)
### END OF SCRIPT
#PlayGame()

all_rewards = 0
qNetwork = QValueNetwork(3,2)
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



            