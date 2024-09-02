
# Documentação da Rede Neural

## Introdução
Esta documentação descreve o funcionamento de uma rede neural simples implementada em JavaScript. A rede neural é projetada para resolver o problema do operador lógico XOR. A implementação inclui as operações básicas de feedforward e backpropagation, bem como a função de ativação sigmoide.

## Estrutura da Rede Neural

A rede neural é composta por três camadas:
1. **Camada de Entrada**: Recebe os dados de entrada.
2. **Camada Oculta**: Processa as entradas usando pesos e bias.
3. **Camada de Saída**: Gera a saída final da rede.

## Classe `NeuralNetwork`

### Construtor

```javascript
constructor(inputNodes, hiddenNodes, outputNodes)
```

O construtor inicializa a rede neural com:
- `inputNodes`: Número de nós na camada de entrada.
- `hiddenNodes`: Número de nós na camada oculta.
- `outputNodes`: Número de nós na camada de saída.

Além disso, o construtor:
- Inicializa os pesos (`weightsIH` e `weightsHO`) com valores aleatórios entre -1 e 1.
- Inicializa os bias (`biasH` e `biasO`) com valores aleatórios entre -1 e 1.
- Define a taxa de aprendizagem (`learningRate`).

### Funções de Ativação

#### Sigmoid

```javascript
sigmoid(x)
```
Aplica a função sigmoide, que mapeia o valor de entrada para um intervalo entre 0 e 1.

#### Derivada da Sigmoid

```javascript
dsigmoid(y)
```
Calcula a derivada da função sigmoide, necessária para o cálculo do gradiente durante o treinamento.

### Feedforward

```javascript
predict(inputArray)
```
Calcula a saída da rede neural para uma dada entrada. O processo envolve:
- Multiplicação das entradas pelos pesos da camada de entrada para a camada oculta, somando o bias e aplicando a função de ativação sigmoide.
- Multiplicação dos valores da camada oculta pelos pesos da camada oculta para a camada de saída, somando o bias e aplicando novamente a função de ativação.

### Treinamento

```javascript
train(inputArray, targetArray)
```
Treina a rede neural ajustando os pesos e bias com base nos erros calculados. O processo inclui:
- Feedforward para calcular a saída atual da rede.
- Cálculo do erro entre a saída atual e a saída desejada (target).
- Ajuste dos pesos e bias usando backpropagation e a derivada da função sigmoide.

## Exemplo de Uso

A seguir, um exemplo de uso da rede neural para treinar e testar o operador XOR.

```javascript
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
```

Neste exemplo, a rede neural é treinada para reconhecer o operador XOR usando os dados de treinamento fornecidos. Após o treinamento, a rede é testada com todas as possíveis entradas do XOR para verificar sua precisão.

## Conclusão

Esta rede neural simples demonstra os conceitos básicos de redes neurais, incluindo feedforward, backpropagation e ajuste de pesos. Embora seja um exemplo básico, ele serve como uma boa introdução para redes neurais e aprendizado de máquina.
