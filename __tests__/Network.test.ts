import Network from "../Network";
import { Matrix } from "ml-matrix";

describe("Network", () => {
  it("can create matrix", () => {
    const network = new Network([5, 3, 10], 0.1);

    expect(network.biases.length).toBe(2);
    expect(network.biases[0].rows).toBe(3);
    expect(network.biases[0].columns).toBe(1);
    expect(network.biases[1].rows).toBe(10);
    expect(network.biases[1].columns).toBe(1);

    expect(network.weights.length).toBe(2);
    expect(network.weights[0].rows).toBe(5);
    expect(network.weights[0].columns).toBe(3);
    expect(network.weights[1].rows).toBe(3);
    expect(network.weights[1].columns).toBe(10);
  });

  it("can feed forward", () => {
    const network = new Network([5, 3, 10], 0.1);

    const inputs = new Matrix([[1, 2, 3, 4, 5]]);

    const output = network.feedforward(inputs);

    console.log(output);
    expect(output.rows).toBe(1);
    expect(output.columns).toBe(10);
  });
});
