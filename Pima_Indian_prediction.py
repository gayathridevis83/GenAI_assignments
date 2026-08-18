# write a python program to draw the neural network for the the pima indians diabetes prediction problem which was discussed in the class

import matplotlib.pyplot as plt

def draw_neural_network():
    fig, ax = plt.subplots(figsize=(12, 8))

    # Layer definition
    layers = [8, 6, 4, 1]  # Input, Hidden1, Hidden2, Output
    layer_names = ["Input Layer", "Hidden Layer 1",
                   "Hidden Layer 2", "Output Layer"]

    neuron_radius = 0.15
    layer_spacing = 2.5

    neuron_positions = []

    # Draw neurons
    for layer_idx, num_neurons in enumerate(layers):
        x = layer_idx * layer_spacing
        layer_pos = []

        y_offset = (num_neurons - 1) / 2

        for neuron_idx in range(num_neurons):
            y = y_offset - neuron_idx

            circle = plt.Circle(
                (x, y),
                neuron_radius,
                color='skyblue',
                ec='black'
            )

            ax.add_patch(circle)
            layer_pos.append((x, y))

        neuron_positions.append(layer_pos)

        ax.text(
            x,
            y_offset + 1,
            layer_names[layer_idx],
            ha='center',
            fontsize=11,
            fontweight='bold'
        )

    # Draw connections
    for layer_idx in range(len(layers) - 1):
        for x1, y1 in neuron_positions[layer_idx]:
            for x2, y2 in neuron_positions[layer_idx + 1]:
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    'gray',
                    linewidth=0.6
                )

    # Input labels
    input_features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DPF",
        "Age"
    ]

    for i, feature in enumerate(input_features):
        x, y = neuron_positions[0][i]
        ax.text(
            x - 0.6,
            y,
            feature,
            ha='right',
            fontsize=9
        )

    # Output label
    x, y = neuron_positions[-1][0]
    ax.text(
        x + 0.6,
        y,
        "Diabetes\n(0 / 1)",
        fontsize=10,
        fontweight='bold'
    )

    ax.set_xlim(-2, 9)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.title(
        "Pima Indians Diabetes Prediction Neural Network",
        fontsize=18,
        fontweight='bold'
    )

    plt.show()

draw_neural_network()
