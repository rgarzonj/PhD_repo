# PhD_repo
## Tests with MonteCarlo search algorithm
Basic MonteCarlo search algorithms as described in page 673  
http://www.jair.org/papers/paper3484.html

### Requirements  
git clone https://github.com/openai/gym  
cd gym  
pip install -e . # minimal install  


### Basic MC search, linear f. approximation (MountainCar)
QValueFunction uses linear approximation of the Q function 
applied to Mountain Car problem  
- mc_search.py (please run this file)
- QValueFunction.py
- util.py

### Basic MC search, theanets NN as Q f. approximation (MountainCar)
QValueFunction uses Theanets Regressor as function approximator
applied to Mountain Car problem  
- mc_search_theanets.py (please run this file)
- QNetwork_theanets.py
- util.py

### Baseline (Pendulum)  
Reference benchmark using random choice on every step.

### Basic MC search, theanets NN as Q f. approximation (Pendulum)
QValueFunction uses Theanets Regressor as function approximator
applied to Pendulum problem  
- Pend_mc_search_theanets.py (please run this file)
- Pend_QNetwork_theanets.py
- util.py

