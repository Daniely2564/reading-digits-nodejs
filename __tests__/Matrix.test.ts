import { Matrix } from "ml-matrix";

describe("Working with Matrix", () => {
  it("can create matrix", () => {
    const m = new Matrix([
      [1, 2, 3],
      [3, 4, 5],
    ]);
    expect(m.rows).toBe(2);
    expect(m.columns).toBe(3);
  });

  it("can do a dot product", () => {
    const m1 = new Matrix([[3, 4]]);
    const m2 = new Matrix([[2], [1]]);
    const dotOutput = m1.mmul(m2);

    expect(dotOutput.rows).toBe(1);
    expect(dotOutput.columns).toBe(1);
    expect(dotOutput.get(0, 0)).toBe(10);
  });

  it("can do a dot product", () => {
    const m1 = new Matrix([[1, 2, 3, 4]]);
    const m2 = new Matrix([[4], [3], [2], [1]]);
    const dotOutput = m1.mmul(m2);

    expect(dotOutput.rows).toBe(1);
    expect(dotOutput.columns).toBe(1);
    expect(dotOutput.get(0, 0)).toBe(4 + 6 + 6 + 4);
  });

  it("can add", () => {
    const m1 = new Matrix([[3, 4]]);
    const item = new Matrix([[1, 1]]);
    const result = m1.add(item);
    expect(result.get(0, 0)).toBe(4);
    expect(result.get(0, 1)).toBe(5);
  });

  it("can multiply", () => {
    const m1 = new Matrix([[3, 4]]);
    const result = m1.multiply(2);
    expect(result.get(0, 0)).toBe(6);
    expect(result.get(0, 1)).toBe(8);
  });
});
