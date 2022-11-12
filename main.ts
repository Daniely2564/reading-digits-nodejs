import DataMatrixParser from "./DataMatrixParser";
import DataParser from "./DataParser";
import Network from "./Network";

async function main() {
  const dp = new DataParser("./datasets");
  const dmp = new DataMatrixParser(dp);
  const td = await dmp.testingData(true);
  const network = new Network([784, 15, 10], 0.1);
  //   const res = network.feedforward(data[0]);
  network.SGD(td, 1, 50);
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
