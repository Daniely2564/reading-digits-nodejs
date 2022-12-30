import Matrix from "ml-matrix";
import DataParser from "./DataParser";

type MatrixDataTuple = [Matrix, Matrix];

class DataMatrixParser {
  private dataParser: DataParser;
  constructor(dataParser: DataParser) {
    this.dataParser = dataParser;
  }

  /**
   * Returns the in a form that is [[image pixels],[correct classification]]
   * image pixel will have a shape (n x n)
   * classification pixel will have a shape (1, m) m = number of classification
   *
   */
  async trainingData(randomize = false) {
    const matrixDataTuples: MatrixDataTuple[] = [];
    const trainingData = await this.dataParser.getTrainingData();
    for (const num in trainingData) {
      const output = new Array(10).fill(0);
      output[parseInt(num)] = 1;
      for (const greyScale of trainingData[num]) {
        matrixDataTuples.push([
          new Matrix([greyScale]).transpose(),
          new Matrix([output]).transpose(),
        ]);
      }
    }
    if (randomize) {
      matrixDataTuples.sort((a, b) => 0.5 - Math.random());
    }
    return matrixDataTuples;
  }

  async testingData(randomize = false) {
    const matrixDataTuples: MatrixDataTuple[] = [];
    const testingData = await this.dataParser.getTestingData();
    for (const num in testingData) {
      const output = new Array(10).fill(0);
      output[parseInt(num)] = 1;
      for (const greyScale of testingData[num]) {
        matrixDataTuples.push([
          new Matrix([greyScale]).transpose(),
          new Matrix([output]).transpose(),
        ]);
      }
    }
    if (randomize) {
      matrixDataTuples.sort((a, b) => 0.5 - Math.random());
    }
    return matrixDataTuples;
  }
}

export default DataMatrixParser;
