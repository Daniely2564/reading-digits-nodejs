import * as fs from "fs";
import { join } from "path";
import getPixels from "get-pixels";

class DataParser {
  trainingPath: string;
  testingPath: string;
  trainingDirs: string[];
  testingDirs: string[];

  // trainingData: Record<string, number[][]>;
  // testingData: Record<string, number[][]>;
  /**
   * Expects to have two folders, 'testing' and 'training'
   * @param path path to the folder
   */
  constructor(path: string) {
    this.trainingPath = join(path, "training");
    this.testingPath = join(path, "testing");

    this.trainingDirs = fs.readdirSync(this.trainingPath);
    this.testingDirs = fs.readdirSync(this.testingPath);
  }

  async getTrainingData() {
    const data: Record<string, number[][]> = {};
    for (let num of this.trainingDirs) {
      const pathToData = join(this.trainingPath, num);
      const images = fs.readdirSync(pathToData);

      for (const img of images) {
        const pathToImage = join(pathToData, img);

        const greyScale = await readInputToGreyScale(pathToImage);
        if (data[num]) data[num].push(greyScale);
        else {
          data[num] = [];
          data[num].push(greyScale);
        }
      }
    }
    return data;
  }
}

function readInputToGreyScale(path: string): Promise<number[]> {
  return new Promise((resolve, reject) => {
    getPixels(path, (err, { data }) => {
      if (err) return reject(err);
      const greyScale: number[] = [];
      const pixelsToRead = Array.from(Buffer.from(data));
      for (let i = 0; i < pixelsToRead.length / 4; i++) {
        const [r, g, b, a] = pixelsToRead.slice(i * 4, (i + 1) * 4);
        greyScale.push(Math.ceil(0.2126 * r + 0.7152 * g + 0.0722 * b));
      }
      resolve(greyScale);
    });
  });
}

// Grayscale = 0.2126*R + 0.7152*G + 0.0722*B
function convertToArray(buffer: Buffer) {
  const arr = new Array(28 * 28).fill(0);
  let i = 0;
  for (let b of buffer) {
    arr[i++] = b;
  }
  console.log("New --- \n\n\n");
  for (let i = 0; i < 28; i++) {
    console.log(
      arr
        .slice(i * 28, (i + 1) * 28)
        .map((num) => " ".repeat(4 - String(num).length) + String(num))
        .join("")
    );
  }
  return arr;
}

export default DataParser;
