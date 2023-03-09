import neat

# Input and output data for the XOR problem
xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_outputs = [0, 1, 1, 0]

# Fitness function definition
def eval_xor_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0  # Starting fitness value
        net = neat.nn.FeedForwardNetwork.create(genome, config)  # Creating the neural network from genome
        for xi, xo in zip(xor_inputs, xor_outputs):  # Iterating over the XOR input-output pairs
            output = net.activate(xi)  # Calculating the neural network output for the current input
            genome.fitness -= (output[0] - xo) ** 2  # Decreasing the fitness value according to the error between output and target

# Defining NEAT configuration and neural network architecture
config_path = "config-feedforward.txt"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
p = neat.Population(config)

# Training
winner = p.run(eval_xor_genomes)

# Printing the result
print('\nBest genome:\n{!s}'.format(winner))
