export const randn = () => {
  let u = 1 - Math.random(); //Converting [0,1) to (0,1)
  let v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};
