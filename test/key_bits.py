import logging
import argparse
import ast
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
import sys
sys.path.append("../src")
import vital_resource.networks as networks

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    outputDirectory,
    numberOfBits,
    keyBits,
    numberOfEpochs,
    architecture,
    numberOfExamples,
    validationRatio,
    rewardMultiplier
):
    logging.info(f"key_bits.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    neural_net = None
    arch_tokens = architecture.split('_')
    if arch_tokens[0] == 'HungryLayerStack':
        neural_net = networks.HungryLayerStack(
            number_of_inputs=int(arch_tokens[1]),
            layer_widths=ast.literal_eval(arch_tokens[2]),
            threshold_range=ast.literal_eval(arch_tokens[3]),
            resource=float(arch_tokens[4]),
            activation_cost=float(arch_tokens[5]),
            training=True
        )
    elif arch_tokens[0] == 'HLSEnsemble':
        neural_net = networks.HLSEnsemble(
            number_of_networks=int(arch_tokens[1]),
            number_of_inputs=int(arch_tokens[2]),
            layer_widths=ast.literal_eval(arch_tokens[3]),
            threshold_range=ast.literal_eval(arch_tokens[4]),
            resource=float(arch_tokens[5]),
            activation_cost=float(arch_tokens[6]),
            training=True
        )
    else:
        raise NotImplementedError(f"key_bits.main(): Not implemented architecture '{architecture}'")

    # Generate random codes
    dataset = generate_random_codes(numberOfExamples, numberOfBits, keyBits)
    logging.debug(f"dataset:\n{dataset}")
    number_of_positives = number_of_positive_examples(dataset)
    logging.debug(f"number_of_positives = {number_of_positives}/{len(dataset)}")

    X = [x for (x, y) in dataset]
    y = [y for (x, y) in dataset]
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=validationRatio, shuffle=True, stratify=y
    )
    logging.debug(f"X_train:\n{X_train}")
    logging.debug(f"y_train:\n{y_train}")
    logging.debug(f"X_validation:\n{X_validation}")
    logging.debug(f"y_validation:\n{y_validation}")

    """random.shuffle(dataset)

    number_of_validation_examples = round(validationRatio * len(dataset))
    validation_dataset = dataset[0: number_of_validation_examples]
    training_dataset = dataset[number_of_validation_examples:]
    """

    with open(os.path.join(outputDirectory, "epoch_log.csv"), 'w') as epoch_log_file:
        epoch_log_file.write(f"epoch,training_accuracy,validation_accuracy,number_of_live_neurons\n")
        for epoch in range(1, numberOfEpochs + 1):
            logging.info(f" ---- Epoch {epoch} ----")
            neural_net.set_training(True)
            with open(os.path.join(outputDirectory, f"epoch{epoch}_log.csv"), 'w') as examples_log_file:
                examples_log_file.write(f"example,avg_rsrc1,avg_rsrc2,dead_neurons1,dead_neurons2\n")
                for example_ndx in range(len(X_train)):
                    #inputs, target_output = training_dataset[example_ndx]
                    inputs = X_train[example_ndx]
                    target_output = y_train[example_ndx]
                    outputs = neural_net.activate(inputs)
                    prediction = neural_net.likelyhood() >= 0.5
                    if prediction == target_output:
                        #logging.info(f" ++++ Reward! ++++ prediction = {prediction}; target_output = {target_output}")
                        neural_net.reward(target_output, rewardMultiplier)
                    #number_of_dead_neurons_list = neural_net.prune_dead_neurons()
                    #logging.info(f"number_of_dead_neurons_list = {number_of_dead_neurons_list}")
                    average_resources = neural_net.average_resources()
                    examples_log_file.write(f"{example_ndx},{average_resources[0]}\n") #,{average_resources[1]}\n")#,{number_of_dead_neurons_list[0]},{number_of_dead_neurons_list[1]}\n")

            number_of_dead_neurons_list = neural_net.prune_dead_neurons()
            logging.debug(f"number_of_dead_neurons_list = {number_of_dead_neurons_list}")
            #neural_net.display()
            """if epoch % 20 == 0:
                logging.info(f" ++++ Updating weights ++++")
                neural_net.update_weights()
            """
            number_of_live_neurons = neural_net.number_of_live_neurons()
            neural_net.set_training(False)

            # Training accuracy
            number_of_correct_training_predictions = 0
            for training_example, training_target_output in zip(X_train, y_train):
                training_outputs = neural_net.activate(training_example)
                prediction = neural_net.likelyhood() >= 0.5
                if prediction == training_target_output:
                    number_of_correct_training_predictions += 1
            training_accuracy = number_of_correct_training_predictions / len(X_train)

            # Validation
            number_of_correct_validation_predictions = 0
            for validation_example, validation_target_output in zip(X_validation, y_validation):
                validation_outputs = neural_net.activate(validation_example)
                prediction = neural_net.likelyhood() >= 0.5
                if prediction == validation_target_output:
                    number_of_correct_validation_predictions += 1
            validation_accuracy = number_of_correct_validation_predictions/len(X_validation)
            logging.info(f"training_accuracy = {training_accuracy}; validation_accuracy = {validation_accuracy}; number_of_live_neurons = {number_of_live_neurons}")

            #average_resources = neural_net.average_resources()
            epoch_log_file.write(f"{epoch},{training_accuracy},{validation_accuracy},{number_of_live_neurons}\n")

def random_code(number_of_bits, key_bits):
    code = np.random.randint(0, 2, number_of_bits)
    if key_bits is not None:
        for k, v in key_bits.items():
            code[k] = v
    return code

def is_positive_example(code, key_bits):
    for k, v in key_bits.items():
        if code[k] != v:
            return False
    return True

def generate_random_codes(number_of_examples, number_of_bits, key_bits):
    examples = [random_code(number_of_bits, key_bits) for _ in range(number_of_examples // 2)]
    examples += [random_code(number_of_bits, None) for _ in range(number_of_examples // 2)]
    dataset = [(c, is_positive_example(c, key_bits)) for c in examples]
    target_number_of_positives = number_of_examples//2
    while number_of_positive_examples(dataset) > target_number_of_positives:
        dataset.pop(0)  # Remove a positive example
        code = random_code(number_of_bits, None)
        dataset.append((code, is_positive_example(code, key_bits)))
    return dataset

def number_of_positive_examples(dataset):
    positives = 0
    for c_isPositive in dataset:
        if c_isPositive[1] == True:
            positives += 1
    return positives

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_key_bits'", default='./output_key_bits')
    parser.add_argument('--numberOfBits', help="The number of bits in the code. Default: 20", type=int, default=20)
    parser.add_argument('--keyBits', help="The dictionary of key bits. Default: '{0: 1, 3: 0, 7: 1}'", default='{0: 1, 3: 0, 7: 1}')
    parser.add_argument('--numberOfEpochs', help="The number of epochs. Default: 100", type=int, default=100)
    parser.add_argument('--architecture', help="The neural network architecture. Default: 'HungryLayerStack_20_[5,5]_[-3,3]_0.5_0.1'", default='HungryLayerStack_20_[5,5]_[-3,3]_0.5_0.1')
    parser.add_argument('--numberOfExamples', help="The number of training examples. Default: 128", type=int, default=128)
    parser.add_argument('--validationRatio', help="The proportion of examples for validation. Default: 0.2", type=float, default=0.2)
    parser.add_argument('--rewardMultiplier', help="The reward multiplier (1.0 covers the cost of activation). Default: 1.5", type=float, default=1.5)
    args = parser.parse_args()
    args.keyBits = ast.literal_eval(args.keyBits)

    main(
        args.outputDirectory,
        args.numberOfBits,
        args.keyBits,
        args.numberOfEpochs,
        args.architecture,
        args.numberOfExamples,
        args.validationRatio,
        args.rewardMultiplier
    )