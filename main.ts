import DataMatrixParser from "./DataMatrixParser";
import DataParser from "./DataParser";
import Network from "./Network";

async function main() {
  const dp = new DataParser("./datasets");
  const dmp = new DataMatrixParser(dp);
  const td = await dmp.testingData();
  // const network = new Network([784, 36, 10], 0.1); // Initial way if no previous learning
  const network = Network.load("test-log-with-training-data.json");
  // network.SGD(td, 30, 50);
  console.log(network.evaluate(td));
}

/**
 *  const zeroSample = trainingData["0"][1];
  for (let i = 0; i < zeroSample.length / 28; i++) {
    console.log(
      zeroSample
        .slice(i * 28, (i + 1) * 28)
        .map((num) => " ".repeat(4 - String(num).length) + String(num))
        .join("\n")
    );
  }
 */

main();
