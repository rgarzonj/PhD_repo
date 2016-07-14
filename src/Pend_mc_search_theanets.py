import util
import numpy as np
import gym
import copy
import random
import time
from Pend_QNetwork_theanets import QValueNetwork
from util import saveContents
from util import loadContents

#from skdata.tests.test_caltech import counts_101
#from joblib import Parallel, delayed

MAX_GAME_STEPS = 200
MAX_SIM_STEPS = 20
NUM_ROLLOUTS = 20
EPSILON = 0.1
MAX_EPISODES = 20


P_MIN = -1.2;
P_MAX = 0.5;
V_MIN = -0.07;
V_MAX = 0.07;
P_DIF = P_MAX - P_MIN
V_DIF = V_MAX-V_MIN

t = 0

possibleActions = np.linspace(-2,2,5)
indexs = np.arange(0,len(possibleActions),1)

#Simulate the game starting from 'start_state'    
def SimulateGame(env,start_state,qNetwork,nSimulation,eps):
    #print 'Starting simulation'
    #print start_state
    #qNetwork.resetTrainSet()
    global t
    observedUtility = 0
    currentState = start_state
    for u in range(MAX_SIM_STEPS):
        if (util.flipCoin(eps)):
            #print 'Choosing random action'
            nextActionIndex = random.choice(indexs)
        else:
            #print 'Choosing best action value action'
            qValues = []
            for a in possibleActions:
#                print 'Evaluating action %d' %a
                qValue = qNetwork.computeQFunction (a,currentState)
                qValues.append(qValue)
            #print qValues
            # The following lines would configure that if all qValues have the same value, we
            # do not take accelerate
            #if (qValues[0] == qValues[1] and qValues[1]==qValues[2]):
            #    nextAction = 1
            #else:
            nextActionIndex = qValues.index(max(qValues))
        if (u == 0):
            a_t = nextActionIndex
        #Execute nextAction
        #print 'Simulation executing action %f' %nextAction
        #Commenting the following line to save computing time
        #env.render()
        currentState, reward, done, info = env.step(np.array([possibleActions[nextActionIndex]]))
        observedUtility += reward
        #print currentState
        #time.sleep(0.5)
    #Update parameters w of Q(s,a)
    #qNetwork.resetTrainSet()
    qNetwork.collectSample(observedUtility, possibleActions[a_t], start_state)
    qNetwork.trainNetwork()
    print 'Simulation %d returning %f with action-value %f' %(nSimulation,possibleActions[a_t] , observedUtility)
    #env.render(close=True)
    #env.close()
    # Return action and observed utility
    return a_t, observedUtility

def PlayGame():
    global t    
    #Initialize game state to fixed starting state
    envir = gym.make('Pendulum-v0')    
    nextState = envir.reset()
    totalReward = 0
    #for t = 1 . . . T do
#   qNetwork = QValueNetwork(envir.action_space.n,nextState.shape[0])
    for t in range(MAX_GAME_STEPS):
        rewards = np.zeros(len(possibleActions))
        counting = np.zeros(len(possibleActions))
        #Run N simulated games for i = 1 . . . N do    
        # Initialize QValueNetwork
        qNetwork = QValueNetwork(1,nextState.shape[0])
        #Save the environment as we will need to run rollouts from the current step
        #envir.render(close=True)
        saveContents(envir,'frozenEnvironment.pckl')
        #envir.render()
        eps = EPSILON
        for i in range(NUM_ROLLOUTS):
            # Maybe we could implement epsilon decay?
            #eps = eps*0.99
            frozenEnv = loadContents('frozenEnvironment.pckl')
            #print 'Starting simulation %d' %i
            a_i,r_i = SimulateGame(frozenEnv,nextState,qNetwork,i,eps)
            #print a_i
            #print r_i
            rewards[a_i] += r_i
            counting[a_i] = counting[a_i]+1
            #print rewards
            #print counting
        #For every action    
        #Compute average observed utility for each action            
        avg_rewards = rewards/counting
        print avg_rewards        
        #Just in case some action was never chosen would give nan, set the action to worst case
        avg_rewards[np.isnan(avg_rewards)] = -100000       
        #Execute selected action in game       
        selectedActionIndex = avg_rewards.argmax() 
        selectedAction = possibleActions[selectedActionIndex]        
        print ('Playing next step %d with action %f +++++++++++++++++++++++++++++++++++++++++++++') %(t,selectedAction)
        #Comment the following line to save computing time
        #envir.render()
        nextState,reward,done,info = envir.step(np.array([selectedAction]))
        #print 'Current situation of the game is'
        #print nextState
        qNetwork.resetNetwork()
        totalReward+=reward
        #if (done):
        #if ((1-nextState[0] <0.001) and (nextState[1]-1<0.001)):
        #    print 'Game reached terminal state in %d with reward %f' %(t,totalReward)
        #    break
    print 'Game reached maximum number of steps; totalReward is %f' %totalReward
    envir.close()
    return (totalReward)
### END OF SCRIPT

all_rewards = 0
for episode in range (MAX_EPISODES):
    print '---------------- Starting episode %d ----------------' %episode
    rew = PlayGame()
    all_rewards += rew

avg_rew = all_rewards/float(MAX_EPISODES)
print 'Average reward is %f ' %avg_rew
    



            