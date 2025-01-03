import logging
import numpy as np
import sys
sys.path.append("../src")
import vital_resource.blocks as blocks
import vital_resource.networks as networks

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.debug("test_HungryLayerStack.main()")

    hlse = networks.HLSEnsemble(3, 2, [2], [-1, 1])
    """hls.layers[0].neurons[0].weights = [1, -1]
    hls.layers[0].neurons[0].threshold = 0
    hls.layers[0].neurons[1].weights = [0, 0]
    hls.layers[0].neurons[1].threshold = 1
    """
    hlse.display()
    inputs = [1, 1]
    outputs = hlse.activate(inputs)
    logging.debug(f"outputs = {outputs}")
    likelyhood = hlse.likelyhood()
    logging.debug(f"likelyhood = {likelyhood}")

    # Display resources
    average_resources_before = hlse.average_resources()
    hlse.reward(1, 1)
    average_resources_after = hlse.average_resources()
    logging.info(f"average_resources_before = {average_resources_before}")
    logging.info(f"average_resources_after = {average_resources_after}")

if __name__ == '__main__':
    main()