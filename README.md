# PhD_repo
## Tests with MonteCarlo search algorithm
Basic MonteCarlo search algorithms as described in page 673  
http://www.jair.org/papers/paper3484.html

### Basic MC search, linear f. approximation
QValueFunction uses linear approximation of the Q function 
applied to Mountain Car problem  
- mc_search.py (please run this file)
- QValueFunction.py
- util.py

### Basic MC search, theanets NN as Q function approximation
QValueFunction uses Theanets Regressor as function approximator
applied to Mountain Car problem  
- mc_search_theanets.py (please run this file)
- QNetwork_theanets.py
- util.py