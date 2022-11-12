import DataParser from "./DataParser";

async function main() {
  const dp = new DataParser("./datasets");
  const trainingData = await dp.getTrainingData();
  const zeroSample = trainingData["0"][1];
  for (let i = 0; i < zeroSample.length / 28; i++) {
    console.log(
      zeroSample
        .slice(i * 28, (i + 1) * 28)
        .map((num) => " ".repeat(4 - String(num).length) + String(num))
        .join("")
    );
  }
}

main();
