class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        // Inicializar pesos aleatoriamente
        this.weightsIH = Array(this.hiddenNodes).fill().map(() => 
            Array(this.inputNodes).fill().map(() => Math.random() * 2 - 1)
        );
        this.weightsHO = Array(this.outputNodes).fill().map(() => 
            Array(this.hiddenNodes).fill().map(() => Math.random() * 2 - 1)
        );

        // Inicializar biases
        this.biasH = Array(this.hiddenNodes).fill().map(() => Math.random() * 2 - 1);
        this.biasO = Array(this.outputNodes).fill().map(() => Math.random() * 2 - 1);

        // Taxa de aprendizagem
        this.learningRate = 0.1;
    }

    // Função de ativação (sigmoid)
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivada da função sigmoid
    dsigmoid(y) {
        return y * (1 - y);
    }

    // Feedforward
    predict(inputArray) {
        // Camada de entrada para camada oculta
        let hidden = this.weightsIH.map((hiddenWeights, i) => 
            this.sigmoid(hiddenWeights.reduce((sum, weight, j) => sum + weight * inputArray[j], 0) + this.biasH[i])
        );

        // Camada oculta para camada de saída
        let outputs = this.weightsHO.map((outputWeights, i) => 
            this.sigmoid(outputWeights.reduce((sum, weight, j) => sum + weight * hidden[j], 0) + this.biasO[i])
        );

        return outputs;
    }

    // Treinar a rede
    train(inputArray, targetArray) {
        // Feedforward
        let hidden = this.weightsIH.map((hiddenWeights, i) => 
            this.sigmoid(hiddenWeights.reduce((sum, weight, j) => sum + weight * inputArray[j], 0) + this.biasH[i])
        );
        let outputs = this.weightsHO.map((outputWeights, i) => 
            this.sigmoid(outputWeights.reduce((sum, weight, j) => sum + weight * hidden[j], 0) + this.biasO[i])
        );

        // Calcular o erro
        let outputErrors = targetArray.map((target, i) => target - outputs[i]);

        // Calcular gradientes da camada de saída
        let outputGradients = outputs.map((output, i) => 
            outputErrors[i] * this.dsigmoid(output) * this.learningRate
        );

        // Calcular deltas dos pesos da camada oculta para a camada de saída
        this.weightsHO = this.weightsHO.map((weights, i) => 
            weights.map((weight, j) => weight + outputGradients[i] * hidden[j])
        );

        // Ajustar biases da camada de saída
        this.biasO = this.biasO.map((bias, i) => bias + outputGradients[i]);

        // Calcular o erro da camada oculta
        let hiddenErrors = this.weightsHO[0].map((_, i) => 
            this.weightsHO.reduce((sum, weights, j) => sum + weights[i] * outputErrors[j], 0)
        );

        // Calcular gradientes da camada oculta
        let hiddenGradients = hidden.map((h, i) => 
            hiddenErrors[i] * this.dsigmoid(h) * this.learningRate
        );

        // Calcular deltas dos pesos da camada de entrada para a camada oculta
        this.weightsIH = this.weightsIH.map((weights, i) => 
            weights.map((weight, j) => weight + hiddenGradients[i] * inputArray[j])
        );

        // Ajustar biases da camada oculta
        this.biasH = this.biasH.map((bias, i) => bias + hiddenGradients[i]);
    }
}

// Exemplo de uso
const nn = new NeuralNetwork(2, 4, 1);

// Dados de treinamento para o operador XOR
const trainingData = [
    { inputs: [0, 0], targets: [0] },
    { inputs: [0, 1], targets: [1] },
    { inputs: [1, 0], targets: [1] },
    { inputs: [1, 1], targets: [0] }
];

// Treinamento
for (let i = 0; i < 10000; i++) {
    const data = trainingData[Math.floor(Math.random() * trainingData.length)];
    nn.train(data.inputs, data.targets);
}

// Teste
console.log("0 XOR 0 =", nn.predict([0, 0])[0]);
console.log("0 XOR 1 =", nn.predict([0, 1])[0]);
console.log("1 XOR 0 =", nn.predict([1, 0])[0]);
console.log("1 XOR 1 =", nn.predict([1, 1])[0]);