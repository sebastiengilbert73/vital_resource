import logging
import numpy as np
import sys
sys.path.append("../src")
import vital_resource.blocks as blocks
import vital_resource.networks as networks

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.debug("test_HungryLayerStack.main()")

    hls = networks.HungryLayerStack(2, [2], [-1, 1])
    hls.layers[0].neurons[0].weights = [1, -1]
    hls.layers[0].neurons[0].threshold = 0
    hls.layers[0].neurons[1].weights = [0, 0]
    hls.layers[0].neurons[1].threshold = 1
    hls.display()
    inputs = [1, 1]
    outputs = hls.activate(inputs)
    logging.debug(f"outputs = {outputs}")
    likelyhood = hls.likelyhood()
    logging.debug(f"likelyhood = {likelyhood}")

    # Display resources
    for layer_ndx in range(len(hls.layers)):
        for neuron_ndx in range(len(hls.layers[layer_ndx].neurons)):
            print(f"{hls.layers[layer_ndx].neurons[neuron_ndx].resource} ", end='')
        print()

    hls.reward(0, 1)
    for layer_ndx in range(len(hls.layers)):
        for neuron_ndx in range(len(hls.layers[layer_ndx].neurons)):
            print(f"{hls.layers[layer_ndx].neurons[neuron_ndx].resource} ", end='')
        print()


if __name__ == '__main__':
    main()