import random
import math

#
# Shorthand:
#   "pd_" as a variable prefix means "partial derivative"
#   "d_" as a variable prefix means "derivative"
#   "_wrt_" is shorthand for "with respect to"
#   "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively

class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hiddenLayers, num_hidden_neuron, num_outputs_neuron):
        self.num_inputs = num_inputs
        self.num_hiddenLayers = num_hiddenLayers
        self.num_hidden_neuron = num_hidden_neuron
        self.num_outputs_neuron = num_outputs_neuron

        self.hidden_layer = []
        for hNum in range(num_hiddenLayers):
            self.hidden_layer.append(NeuronLayer(num_hidden_neuron[hNum]))
        self.output_layer = NeuronLayer(num_outputs_neuron)

        self.init_weights_from_inputs_to_hidden_layer_neurons()
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons()

    def init_weights_from_inputs_to_hidden_layer_neurons(self):
        for hNum in range(self.num_hiddenLayers):
            for h in range(self.num_hidden_neuron[hNum]):
                if hNum==0:
                    for i in range(self.num_inputs):
                        self.hidden_layer[hNum].neurons[h].weights.append(random.random())
                else:
                    for i in range(len(self.hidden_layer[hNum-1].neurons)):
                        self.hidden_layer[hNum].neurons[h].weights.append(random.random())

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self):
        for o in range(self.num_outputs_neuron):
            for h in range(self.num_hidden_neuron[self.num_hiddenLayers-1]):
                self.output_layer.neurons[o].weights.append(random.random())

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layers: ', len(self.hidden_layer))
        for h in range(len(self.hidden_layer)):
            print('------')
            print('Hidden Layer', h)
            print('  Bias:', self.hidden_layer[h].bias)
            self.hidden_layer[h].inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        layerInputs = inputs;
        for hNum in range(self.num_hiddenLayers):
            hidden_layer_outputs = self.hidden_layer[hNum].feed_forward(layerInputs)
            layerInputs = hidden_layer_outputs
        return self.output_layer.feed_forward(layerInputs)

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):##
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        for o in range(self.num_outputs_neuron):

            # ∂E/∂zⱼ
           self.output_layer.neurons[o].delta = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        for hNum in reversed(range(self.num_hiddenLayers)):
            if hNum==self.num_hiddenLayers-1:
                for h in range(self.num_hidden_neuron[hNum]):

                    # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
                    # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
                    d_error_wrt_hidden_neuron_output = 0
                    for o in range(self.num_outputs_neuron):
                        d_error_wrt_hidden_neuron_output += self.output_layer.neurons[o].delta * self.output_layer.neurons[o].weights[h]

                    # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
                    self.hidden_layer[hNum].neurons[h].delta = d_error_wrt_hidden_neuron_output * self.hidden_layer[hNum].neurons[h].calculate_pd_total_net_input_wrt_input()        
            else:
                for h in range(self.num_hidden_neuron[hNum]):

                    # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
                    # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
                    d_error_wrt_hidden_neuron_output = 0
                    for o in range(self.num_hidden_neuron[hNum+1]):
                        d_error_wrt_hidden_neuron_output += self.hidden_layer[hNum+1].neurons[o].delta * self.hidden_layer[hNum+1].neurons[o].weights[h]

                    # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
                    self.hidden_layer[hNum].neurons[h].delta = d_error_wrt_hidden_neuron_output * self.hidden_layer[hNum].neurons[h].calculate_pd_total_net_input_wrt_input()

        # 3. Update output neuron weights
        for o in range(self.num_outputs_neuron):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = self.output_layer.neurons[o].delta * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for hNum in reversed(range(self.num_hiddenLayers)):
            for h in range(self.num_hidden_neuron[hNum]):
                for w_ih in range(len(self.hidden_layer[hNum].neurons[h].weights)):

                    # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                    pd_error_wrt_weight = self.hidden_layer[hNum].neurons[h].delta * self.hidden_layer[hNum].neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                    # Δw = α * ∂Eⱼ/∂wᵢ
                    self.hidden_layer[hNum].neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

class NeuronLayer:
    def __init__(self, num_neurons):

        # Every neuron in a layer shares the same bias
        self.bias = random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print(' Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print('  Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('   Weight', w, ':', self.neurons[n].weights[w])

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):#don't know
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.delta = 0

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # Apply the logistic function to squash the output of the neuron
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

# Blog post example:
nn = NeuralNetwork(2, 4, [2, 2, 3, 4], 2)
nn.inspect();
errorTolerance = 0.002
numIter = 10000;
for i in range(numIter):
    nn.train([0.05, 0.1], [0.01, 0.99])
    error = round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9)
    if error<errorTolerance or i==numIter-1:
        print("Total training error =", error)
        break
