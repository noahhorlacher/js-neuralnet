export function sigmoid(x, derivative) {
    let fx = 1 / (1 + Math.exp(-x))
    return derivative ? fx * (1 - fx) : fx
}