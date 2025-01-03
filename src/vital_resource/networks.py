import numpy as np
import vital_resource.blocks as blocks

class HungryLayerStack:
    def __init__(self, number_of_inputs, layer_widths, threshold_range=[-3, 3], resource=0.5,
                 activation_cost=0.1, training=True, replace_with_random_neurons=False):
        self.number_of_inputs = number_of_inputs
        self.layers = []
        self.layers.append(blocks.HungryLayer(number_of_inputs, layer_widths[0], threshold_range, resource, activation_cost, training))
        for layer_ndx in range(1, len(layer_widths)):
            self.layers.append(blocks.HungryLayer(layer_widths[layer_ndx - 1], layer_widths[layer_ndx], threshold_range, resource, activation_cost, training) )
        self.state = None
        self.training = training
        self.replace_with_random_neurons = replace_with_random_neurons

    def activate(self, inputs):
        act = inputs
        for layer_ndx in range(len(self.layers)):
            act = self.layers[layer_ndx].activate(act)
        self.state = act
        return self.state

    def batch_activate(self, inputs_list):
        outputs_list = []
        for input in inputs_list:
            outputs_list.append(self.activate(input))
        return outputs_list

    def likelyhood(self):
        outputs_sum = 0
        number_of_live_outputs = 0
        for state_ndx in range(len(self.state)):
            if self.state[state_ndx] is not None:
                outputs_sum += self.state[state_ndx]
                number_of_live_outputs += 1
        if number_of_live_outputs > 0:
            return outputs_sum/number_of_live_outputs
        else:
            return 0.5

    def reward(self, target_output, multiplier):
        # Last layer
        last_layer_scores = [0] * len(self.layers[-1].neurons)
        for last_layer_neuron_ndx in range(len(self.layers[-1].neurons)):
            if self.layers[-1].neurons[last_layer_neuron_ndx].is_alive and self.state[last_layer_neuron_ndx] == target_output:
                last_layer_scores[last_layer_neuron_ndx] = 1
        self.layers[-1].reward(last_layer_scores, multiplier)

        # Layers L-2, L-3, L-4, ... , 0
        for layer_ndx in range(len(self.layers) - 2, -1, -1):  # [L - 2, L - 3, ..., 0]
            neuron_scores = self.layers[layer_ndx + 1].summarize_input_contributions(neuron_vote_weights=last_layer_scores)
            #print(f"HungryLayerStack.reward(): layer_ndx = {layer_ndx}: neuron_scores = {neuron_scores}")
            self.layers[layer_ndx].reward(neuron_scores, multiplier)
            #print(f"HungryLayerStack.reward(): After self.layers[layer_ndx].reward(neuron_scores)")
            last_layer_scores = neuron_scores

    def set_training(self, is_training):
        self.training = is_training
        for layer in self.layers:
            layer.set_training(self.training)

    def prune_dead_neurons(self):
        number_of_dead_neurons_list = []
        for layer_ndx in range(len(self.layers)):
            number_of_dead_neurons_list.append(self.layers[layer_ndx].prune_dead_neurons(self.replace_with_random_neurons))
        return number_of_dead_neurons_list

    def update_weights(self, max_ratio_for_minus_1=0.4, min_ratio_for_plus_1=0.6):
        for layer_ndx in range(len(self.layers)):
            self.layers[layer_ndx].update_weights(max_ratio_for_minus_1, min_ratio_for_plus_1)

    def average_resources(self):
        averages = []
        for layer in self.layers:
            averages.append(layer.average_resource())
        return averages

    def display(self):
        for layer_ndx in range(len(self.layers)):
            self.layers[layer_ndx].display()
            print()

    def number_of_live_neurons(self):
        n = 0
        for layer_ndx in range(len(self.layers)):
            n += self.layers[layer_ndx].number_of_live_neurons()
        return n

class HLSEnsemble:
    def __init__(self, number_of_networks, number_of_inputs, layer_widths, threshold_range=[-3, 3], resource=0.5, activation_cost=0.1, training=True):
        self.networks = []
        for net_ndx in range(number_of_networks):
            self.networks.append(HungryLayerStack(number_of_inputs, layer_widths, threshold_range, resource, activation_cost, training))
        self.state = None
        self.training = training

    def activate(self, inputs):
        self.state = []
        for net_ndx in range(len(self.networks)):
            self.networks[net_ndx].activate(inputs)
            self.state.append(self.networks[net_ndx].likelyhood())
        return self.state

    def likelyhood(self):
        outputs_sum = sum(self.state)
        return outputs_sum / len(self.state)

    def reward(self, target_output, multiplier):
        for net_ndx in range(len(self.state)):
            prediction = 0
            if self.state[net_ndx] >= 0.5:
                prediction = 1
            if prediction == target_output:
                self.networks[net_ndx].reward(target_output, multiplier)

    def set_training(self, is_training):
        self.training = is_training
        for net_ndx in range(len(self.networks)):
            self.networks[net_ndx].set_training(is_training)

    def prune_dead_neurons(self):
        number_of_dead_neurons_list_list = []
        for net_ndx in range(len(self.networks)):
            number_of_dead_neurons_list_list.append(self.networks[net_ndx].prune_dead_neurons())
        return number_of_dead_neurons_list_list

    def display(self):
        for net_ndx in range(len(self.networks)):
            self.networks[net_ndx].display()
            print()

    def average_resources(self):
        averages = []
        for network in self.networks:
            averages.append(network.average_resources())
        return averages

    def number_of_live_neurons(self):
        n = 0
        for net_ndx in range(len(self.networks)):
            n += self.networks[net_ndx].number_of_live_neurons()
        return n