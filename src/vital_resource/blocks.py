import numpy as np
import random

class HungryNeuron():
    def __init__(self, number_of_inputs, threshold_range=[-3, 3], resource=0.5, activation_cost=0.1, training=True):#:, memory_size=20):
        if resource < 0.0 or resource > 1.0:
            raise ValueError(f"HungryNeuron.__init__(): The resource level ({resource}) must be included in [0, 1]")
        self.weights = np.random.randint(-1, 2, size=number_of_inputs)
        self.threshold = random.randint(threshold_range[0], threshold_range[1])
        self.resource = resource
        self.activation_cost = activation_cost
        self.training = training
        self.state = None
        self.last_inputs = None
        self.is_alive = True

    def activate(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError(f"HungryNeuron.activate(): len(inputs) ({len(inputs)}) != len(self.weights) ({len(self.weights)})")
        if self.training and self.is_alive:
            self.resource -= self.activation_cost
        #if self.resource > 0:
        if self.is_alive:
            self.last_inputs = inputs
            inputs = [i if i is not None else 0 for i in inputs]
            #print(f"HungryNeuron.activate(): inputs = {inputs}")
            sum = int(np.dot(self.weights, inputs))
            if sum >= self.threshold:
                self.state = 1
            else:
                self.state = 0
        else:
            self.last_inputs = None
            self.state = None
        return self.state

    def reward(self, resource):
        if self.is_alive:
            self.resource = min(self.resource + resource, 1.0)

    def input_contributed_to_last_activation(self):
        contributions = []
        if self.is_alive:
            for input_ndx in range(len(self.weights)):
                contributed = False
                input = self.last_inputs[input_ndx]
                if self.state == 1:
                    if input == 1 and self.weights[input_ndx] == 1:
                        contributed = True
                    elif input == 0 and self.weights[input_ndx] == -1:
                        contributed = True
                elif self.state == 0:
                    if input == 1 and self.weights[input_ndx] == -1:
                        contributed = True
                    elif input == 0 and self.weights[input_ndx] == 1:
                        contributed = True
                contributions.append(contributed)
        return contributions

    def display(self):
        if not self.is_alive:
            state_str = 'dead'
        else:
            state_str = "W: "
            for weight_ndx in range(len(self.weights)):
                state_str += f"{self.weights[weight_ndx]}, "
            state_str += f"T: {self.threshold}"
        print(state_str)


class HungryLayer():
    def __init__(self, number_of_inputs, number_of_neurons, threshold_range=[-3, 3], resource=0.5, activation_cost=0.1, training=True):
        self.neurons = [HungryNeuron(number_of_inputs, threshold_range, resource, activation_cost, training) for i in range(number_of_neurons)]
        self.layer_activation_cost = number_of_neurons * activation_cost
        self.number_of_inputs = number_of_inputs
        self.threshold_range = threshold_range
        self.initial_resource = resource
        self.neuron_activation_cost = activation_cost
        self.training = training
        self.state = None

    def activate(self, inputs):
        self.state = []
        for neuron_ndx in range(len(self.neurons)):
            self.state.append(self.neurons[neuron_ndx].activate(inputs))
        return self.state

    def reward(self, neuron_scores, multiplier):
        if len(neuron_scores) != len(self.neurons):
            raise ValueError(f"HungryLayer.reward(): len(neuron_scores) ({len(neuron_scores)}) != len(self.neurons) ({len(self.neurons)})")
        # Normalize the scores
        normalized_scores = neuron_scores
        if sum(neuron_scores) > 0:
            normalized_scores = [s/sum(neuron_scores) for s in neuron_scores]
        for neuron_ndx in range(len(self.neurons)):
            if normalized_scores[neuron_ndx] > 0:
                self.neurons[neuron_ndx].reward(multiplier * normalized_scores[neuron_ndx] * self.layer_activation_cost)

    def summarize_input_contributions(self, neuron_vote_weights):
        if len(neuron_vote_weights) != len(self.neurons):
            raise ValueError(f"len(neuron_vote_weights) ({len(neuron_vote_weights)}) != len(self.neurons) ({len(self.neurons)})")
        scores = [0] * self.number_of_inputs
        for neuron_ndx in range(len(self.neurons)):
            vote_weight = neuron_vote_weights[neuron_ndx]
            if vote_weight > 0 and self.neurons[neuron_ndx].is_alive:
                neuron = self.neurons[neuron_ndx]
                input_contributions = neuron.input_contributed_to_last_activation()
                for input_ndx in range(len(input_contributions)):
                    if input_contributions[input_ndx]:
                        scores[input_ndx] += vote_weight
        return scores

    def set_training(self, is_training):
        self.training = is_training
        for neuron in self.neurons:
            neuron.training = self.training

    def prune_dead_neurons(self, replace_with_random_neurons):
        number_of_dead_neurons = 0
        for neuron_ndx in range(len(self.neurons)):
            if self.neurons[neuron_ndx].resource <= 0 and self.neurons[neuron_ndx].is_alive:  # Generate a new random neuron
                self.neurons[neuron_ndx] = HungryNeuron(self.number_of_inputs, self.threshold_range, self.initial_resource,
                                                        self.neuron_activation_cost, self.training)
                if not replace_with_random_neurons:
                    self.neurons[neuron_ndx].is_alive = False
                number_of_dead_neurons += 1
        return number_of_dead_neurons

    def average_resource(self):
        sum = 0
        for neuron in self.neurons:
            sum += neuron.resource
        average = sum/len(self.neurons)
        return average

    def display(self):
        for neuron_ndx in range(len(self.neurons)):
            self.neurons[neuron_ndx].display()

    def number_of_live_neurons(self):
        n = 0
        for neuron_ndx in range(len(self.neurons)):
            if self.neurons[neuron_ndx].is_alive:
                n += 1
        return n