import { Matrix } from "ml-matrix";
import { writeFileSync, readFileSync } from "fs";
import { randn } from "./lib";

(Matrix.prototype as any).shape = function () {
  return [this.rows, this.columns];
};

type Tuple = [number, number];

// [Input, Output]
type Batch = [Matrix, Matrix];

class Network {
  numLayers: number;
  sizes: number[];
  biases: Matrix[];
  weights: Matrix[];
  learningRate: number;

  /**
   * Takes in the number of neurons in each layer including Input and Output layers
   *  along with the hidden. The first item of the array represents Input neurons and
   * the last item of the layer represents the output layers.
   * e.g. [5, 25, 10] - 5 input neurons, 25 hdden neurons in a hidden layer, 10 output neurons
   *
   * @param sizes number[]
   */
  constructor(sizes: number[], learningRate: number) {
    this.numLayers = sizes.length;
    this.sizes = sizes;
    this.learningRate = learningRate;
    this.biases = this.getRandomMatrixFromSize(
      sizes.slice(1).map((size: number) => [size, 1])
    );
    this.weights = this.getRandomMatrixFromSize(
      sizes.slice(1).map((size, index) => [size, sizes[index]])
    );
  }

  private getRandomArrBySize(size: number) {
    return new Array(size).fill(0).map(() => randn());
  }

  private getRandomMatrixFromSize(tuples: Tuple[]) {
    const matrix: Matrix[] = [];

    for (const tuple of tuples) {
      const arr: number[][] = [];
      const [row, col] = tuple;
      for (let i = 0; i < row; i++) {
        arr.push(this.getRandomArrBySize(col));
      }
      matrix.push(new Matrix(arr));
    }

    return matrix;
  }

  feedforward(input: Matrix) {
    let layer = 0;
    let mat = input;
    for (; layer < this.weights.length; layer++) {
      mat = this.sigmoid(this.weights[layer].mmul(mat).add(this.biases[layer]));
    }

    return mat as Matrix;
  }

  /**
   * Stochastic gradient descent
   */
  SGD(
    trainingData: Batch[],
    epochs: number,
    miniBatchSize: number,
    testData?: Batch[]
  ) {
    let noOfTest = 0;
    if (testData) {
      noOfTest = testData.length;
    }
    const size = trainingData.length;

    for (let epoch = 0; epoch < epochs; epoch++) {
      const td = trainingData.sort((a, b) => 0.5 - Math.random());
      for (let i = 0; i < trainingData.length / miniBatchSize; i++) {
        const miniBatch = trainingData.slice(
          i * miniBatchSize,
          (i + 1) * miniBatchSize
        );
        this.updateMiniBatch(miniBatch);
      }
      const evaluation = this.evaluate(trainingData);
      console.log(
        `Finished epoch ${epoch + 1}. The evaluation is : ${evaluation}`
      );
    }
  }

  private updateMiniBatch(batches: Batch[]) {
    let nablaBias = this.biases.map((bias) =>
      Matrix.zeros(bias.rows, bias.columns)
    );
    let nablaWeights = this.weights.map((weight) =>
      Matrix.zeros(weight.rows, weight.columns)
    );
    for (const batch of batches) {
      const [x, y] = batch;
      const [deltaNablaBias, deltaNablaWeights] = this.backprop(x, y);
      nablaBias = nablaBias.map((nb, i) => nb.add(deltaNablaBias[i]));
      nablaWeights = nablaWeights.map((nw, i) => nw.add(deltaNablaWeights[i]));
    }
    this.weights = this.weights.map((w, i) =>
      w.subtract(nablaWeights[i].multiply(this.learningRate / batches.length))
    );
    this.biases = this.biases.map((b, i) =>
      b.subtract(nablaBias[i].multiply(this.learningRate / batches.length))
    );
  }

  private backprop(x: Matrix, y: Matrix): [Matrix[], Matrix[]] {
    const nablaBias = this.biases.map((bias) =>
      Matrix.zeros(bias.rows, bias.columns)
    );
    const nablaWeights = this.weights.map((weight) =>
      Matrix.zeros(weight.rows, weight.columns)
    );
    // Feed Forward
    let activation: Matrix = x;
    const activations: Matrix[] = [x];

    const zs: Matrix[] = []; // List to store all the z vectors
    for (let layer = 0; layer < this.weights.length; layer++) {
      const z = this.weights[layer].mmul(activation).add(this.biases[layer]);
      zs.push(z);

      activation = this.sigmoid(z);
      activations.push(activation);
    }

    // Backward Pass
    let delta = this.costDerivative(
      activations[activations.length - 1],
      y
    ).multiply(this.sigmoidPrime(zs[zs.length - 1]));

    nablaBias[nablaBias.length - 1] = delta;
    nablaWeights[nablaWeights.length - 1] = delta.mmul(
      activations[activations.length - 2].transpose()
    );

    for (let offset = 2; offset < this.numLayers; offset++) {
      const z = zs[zs.length - offset];
      const sp = this.sigmoidPrime(z);
      delta = this.weights[this.weights.length - offset + 1]
        .transpose()
        .mmul(delta)
        .multiply(sp);
      nablaBias[nablaBias.length - offset] = delta;
      nablaWeights[nablaWeights.length - offset] = delta.mmul(
        activations[activations.length - offset - 1].transpose()
      );
    }

    return [nablaBias, nablaWeights];
  }

  private costDerivative(outputActivations: Matrix, y: Matrix) {
    return outputActivations.subtract(y);
  }

  private sigmoidPrime(matrix: Matrix) {
    return this.sigmoid(matrix).multiply(
      Matrix.ones(matrix.rows, matrix.columns).subtract(this.sigmoid(matrix))
    );
  }

  sigmoid(matrix: Matrix) {
    return Matrix.ones(matrix.rows, matrix.columns).div(
      Matrix.ones(matrix.rows, matrix.columns).add(
        Matrix.exp(matrix.multiply(-1))
      )
    );
  }

  evaluate(batches: Batch[]) {
    let corrs = 0;
    for (const [x, y] of batches) {
      const predictedOutput = this.feedforward(x).maxColumnIndex(0)[0];
      const actualOutput = y.maxColumnIndex(0)[0];
      if (predictedOutput === actualOutput) corrs++;
    }

    return corrs / batches.length;
  }

  save(filename: string) {
    const weights = this.weights.map((w) => w.toJSON());
    const biases = this.biases.map((b) => b.toJSON());

    const learningRate = this.learningRate;
    const sizes = this.sizes;
    const layers = this.numLayers;

    writeFileSync(
      `./savedLogs/${filename}`,
      JSON.stringify({
        sizes,
        layers,
        learningRate,
        weights,
        biases,
      })
    );
  }

  static load(filename: string) {
    const data = JSON.parse(
      readFileSync(`./savedLogs/${filename}`, { encoding: "utf-8" })
    );
    const newNetwork = new Network(data.sizes, data.learningRate);
    newNetwork.numLayers = data.layers;
    newNetwork.weights = data.weights.map((weight: any) => new Matrix(weight));
    newNetwork.biases = data.biases.map((bias: any) => new Matrix(bias));

    return newNetwork;
  }
}

export default Network;
