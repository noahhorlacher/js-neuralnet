import { add, subtract, multiply, dotMultiply, mean, abs, transpose, random } from 'mathjs'

export class Network {
    constructor(
        amount_input_neurons,
        amount_hidden_neurons,
        amount_output_neurons,
        epochs,
        activation_function,
        learning_rate,
        update_frequency = 10_000
    ) {
        this.amount_input_neurons = amount_input_neurons
        this.amount_hidden_neurons = amount_hidden_neurons
        this.amount_output_neurons = amount_output_neurons
        this.epochs = epochs
        this.activation_function = activation_function
        this.learning_rate = learning_rate
        this.update_frequency = update_frequency

        // weights for synapses
        // input to hidden
        this.synapses0 = random([this.amount_input_neurons, this.amount_hidden_neurons], -1.0, 1.0)

        // hidden to output
        this.synapses1 = random([this.amount_hidden_neurons, this.amount_output_neurons], -1.0, 1.0)
    }

    train(input_layer, target_values) {
        for (let epoch = 0; epoch < this.epochs; epoch++) {
            // forward
            let hidden_layer = multiply(input_layer, this.synapses0).map(
                synapse_output => this.activation_function(synapse_output, false)
            ) // hidden layer output matrix
            let output_layer = multiply(hidden_layer, this.synapses1).map(
                synapse_output => this.activation_function(synapse_output, false)
            ) // output layer output matrix

            // backtrace
            let output_error = subtract(target_values, output_layer) // calculate output error matrix
            let output_delta = dotMultiply(output_error, output_layer.map(
                synapse_output => this.activation_function(synapse_output, true)
            )) // calculate output delta error vector
            let hidden_error = multiply(output_delta, transpose(this.synapses1)) // calculate hidden error matrix
            let hidden_delta = dotMultiply(hidden_error, hidden_layer.map(
                synapse_output => this.activation_function(synapse_output, true)
            )) // calculate hidden delta error vector

            // gradient descent
            this.synapses1 = add(
                this.synapses1,
                multiply(
                    transpose(hidden_layer),
                    multiply(output_delta, this.learning_rate)
                )
            )
            this.synapses0 = add(
                this.synapses0,
                multiply(
                    transpose(input_layer),
                    multiply(hidden_delta, this.learning_rate)
                )
            )

            // update console
            if (epoch % this.update_frequency === 0) {
                console.log(`Epoch ${epoch}/${this.epochs}`)
                console.log(`Error: ${mean(abs(output_error))}`)
            }
        }
    }

    predict(input_layer) {
        let hidden_layer = multiply(input_layer, this.synapses0).map(synapse_output => this.activation_function(synapse_output, false))
        let output_layer = multiply(hidden_layer, this.synapses1).map(synapse_output => this.activation_function(synapse_output, false))
        return output_layer
    }
}