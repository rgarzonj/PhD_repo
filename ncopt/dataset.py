import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
from scipy.spatial.distance import pdist, squareform 
#from sklearn.decomposition import PCA
# from tsp_with_ortools import Solver



class DataGenerator(object):

    # Initialize a DataGenerator
    def __init__(self,numBlocks):
        self.numBlocks = numBlocks
        #self.solver=solver  # reference solver for TSP (Google_OR_tools)
        with open('input_ncopt_'+ str(self.numBlocks) + ".txt",'r', encoding='utf-8') as f:
            self.instances = f.read().split('\n')
        self.numInstances = len(self.instances)


    # Solve an instance with reference solver
    def solve_instance(self, sequence):

        # Calculate dist_matrix
        dist_array = pdist(sequence)
        dist_matrix = squareform(dist_array)
        # Call OR Tools to solve instance
        route, opt_tour_length=self.solver.run(dist_matrix)
        # Corresponding tour
        ordered_seq = sequence[route]

        return ordered_seq[:-1], opt_tour_length


    # Generate random TSP instance
    def gen_instance_OLD(self, max_length, dimension, test_mode=True, seed=0):
        if seed!=0: np.random.seed(seed)

        # Randomly generate (max_length) cities with (dimension) coordinates in [0,100]
        seq = np.random.randint(self.numInstances)

        # Principal Component Analysis to center & rotate coordinates
        pca = PCA(n_components=dimension)
        sequence = pca.fit_transform(seq)

        # Scale to [0,1[
        input_ = sequence/100

        if test_mode == True:
            return input_, seq
        else:
            return input_

# Generate random BlocksWorld instance
    def gen_instance(self, max_length, dimension, test_mode=True, seed=0):
        
        if seed!=0: np.random.seed(seed)

        # Randomly generate (max_length) cities with (dimension) coordinates in [0,100]
        i = np.random.randint(self.numInstances)
        seq = self.instances[i]
        input_ = seq.split(' output ')[0].split(" ")
        input_int = []
        for one_item in input_:         
            input_int.append(int(one_item))
        input_array = np.array(input_int,dtype="float64")
        if test_mode == True:
            return np.reshape(input_array,(len(input_array),1)), seq
        else:
            return np.reshape(input_array,(len(input_array),1))

    # Generate random batch for training procedure
    def train_batch(self, batch_size, max_length, dimension):
        input_batch = []

        for _ in range(batch_size):
            # Generate random TSP instance
            input_ = self.gen_instance(max_length, dimension, test_mode=False)
            #print (input_)
            #print (input_.shape)
            # Store batch
            input_batch.append(input_)
        #print (len(input_batch))
        return input_batch


    # Generate random batch for testing procedure
    def test_batch(self, batch_size, max_length, dimension, seed=0):
        # Generate random TSP instance
        input_, or_sequence = self.gen_instance(max_length, dimension, test_mode=True, seed=seed)

        # Store batch
        input_batch = np.tile(input_,(batch_size,1,1))

        return input_batch, or_sequence


    # Plot a tour
    def visualize_2D_trip(self, trip):
        plt.figure(figsize=(30,30))
        rcParams.update({'font.size': 22})

        # Plot cities
        plt.scatter(trip[:,0], trip[:,1], s=200)

        # Plot tour
        tour=np.array(list(range(len(trip))) + [0])
        X = trip[tour, 0]
        Y = trip[tour, 1]
        plt.plot(X, Y,"--", markersize=100)

        # Annotate cities with order
        labels = range(len(trip))
        for i, (x, y) in zip(labels,(zip(X,Y))):
            plt.annotate(i,xy=(x, y))  

        plt.xlim(0,100)
        plt.ylim(0,100)
        plt.show()


    # Heatmap of permutations (x=cities; y=steps)
    def visualize_sampling(self, permutations):
        max_length = len(permutations[0])
        grid = np.zeros([max_length,max_length]) # initialize heatmap grid to 0

        transposed_permutations = np.transpose(permutations)
        for t, cities_t in enumerate(transposed_permutations): # step t, cities chosen at step t
            city_indices, counts = np.unique(cities_t,return_counts=True,axis=0)
            for u,v in zip(city_indices, counts):
                grid[t][u]+=v # update grid with counts from the batch of permutations

        # plot heatmap
        fig = plt.figure()
        rcParams.update({'font.size': 22})
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(grid, interpolation='nearest', cmap='gray')
        plt.colorbar()
        plt.title('Sampled permutations')
        plt.ylabel('Time t')
        plt.xlabel('City i')
        plt.show()









if __name__ == "__main__":

    # Config
    numBlocks = 3
    batch_size=128
    max_length=(numBlocks+1)*2
    dimension=1

    # Create Solver and Data Generator
    solver = [] #Solver(max_length)
    dataset = DataGenerator(numBlocks)

    # Generate some data
    input_batch = dataset.train_batch(batch_size, max_length, dimension)
    #input_batch, or_sequence = dataset.test_batch(batch_size, max_length, dimension,seed=1)
    #print (type(input_batch))
    #print (input_batch[0])
    #print (input_batch[0].shape)
    #print (input_batch.shape)
    # Some print
    #print('Input batch: \n',input_batch)
    
    # 2D plot for coord batch
    #dataset.visualize_2D_trip(100*input_batch[0])

    # Solve to optimality and plot solution    
    #opt_trip, opt_length = dataset.solve_instance(or_sequence)
    #print('Solver tour length: \n', opt_length)
    #dataset.visualize_2D_trip(opt_trip)