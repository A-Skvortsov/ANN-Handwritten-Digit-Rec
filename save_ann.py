# Code for saving the ANN (its weights and biases) to external csv files
# Should eventually add load() functionality to load a saved ANN from csv
import csv
EMPTY = "          "  # for formatting purposes

"""
Saving all ANN info to csv for debugging and general analysis
Saves:
    - current weights and biases
    - all current gradients
    - all current neuron values (as a result of last forward pass)
"""
def save():
    with open('0_all_biases.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['B0', 'B1', 'B2', 'B3', '', 'BG0', 'BG1', 'BG2', 'BG3'])
    
        for i in range(max([len(mlp.layers[i].neurons) for i in range(len(mlp.layers))])):  # 'for neuron in layer with most neurons'
            row = [0.0 for j in range(2 * len(mlp.layers) + 1)]
            for j in range(len(mlp.layers)):  # biases
                try: row[j] = mlp.layers[j].neurons[i].bias.val
                except: row[j] = EMPTY
            row[len(mlp.layers)] = EMPTY  # space column for readability
            for j in range(len(mlp.layers) + 1, 2*len(mlp.layers) + 1):  # bias gradients
                try: row[j] = mlp.layers[j - len(mlp.layers) - 1].neurons[i].bias.grad
                except: row[j] = EMPTY
            writer.writerow(row)

    with open('0_all_neurons.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['V0', 'V1', 'V2', 'V3', '', 'VG0', 'VG1', 'VG2', 'VG3'])
    
        for i in range(max([len(mlp.layers[i].neurons) for i in range(len(mlp.layers))])):  # 'for neuron in layer with most neurons'
            row = [0.0 for j in range(2 * len(mlp.layers) + 1)]
            for j in range(len(mlp.layers)):  # values
                try: row[j] = mlp.layers[j].neurons[i].val.val
                except: row[j] = EMPTY
            row[len(mlp.layers)] = EMPTY  # space column for readability
            for j in range(len(mlp.layers) + 1, 2*len(mlp.layers) + 1):  # value gradients
                try: row[j] = mlp.layers[j - len(mlp.layers) - 1].neurons[i].val.grad
                except: row[j] = EMPTY
            writer.writerow(row)

    with open('0_all_weights.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['W0', 'W1', 'W2', 'W3', '', 'WG0', 'WG1', 'WG2', 'WG3'])

        for i in range(max([len(mlp.layers[x].neurons) for x in range(len(mlp.layers))])):  # 'for neuron in layer with most neurons'
            row = [0.0 for j in range(2 * len(mlp.layers) + 1)]
            for w in range(max([len(mlp.layers[x].neurons[0].weights) for x in range(len(mlp.layers))])):  # 'for weight in layer with most weights'
                for j in range(len(mlp.layers)):  # weights
                    try: row[j] = mlp.layers[j].neurons[i].weights[w].val
                    except: row[j] = EMPTY
                row[len(mlp.layers)] = EMPTY  # space column for readability
                for j in range(len(mlp.layers) + 1, 2*len(mlp.layers) + 1):  # weight gradients
                    try: row[j] = mlp.layers[j - len(mlp.layers) - 1].neurons[i].weights[w].grad
                    except: row[j] = EMPTY
                writer.writerow(row)
            writer.writerow([EMPTY for i in range(2 * len(mlp.layers) + 1)])