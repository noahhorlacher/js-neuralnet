import { Network } from './network.js'
import { matrix } from 'mathjs'
import { sigmoid } from './activations.js'

const input = matrix([
    [1],
    [2],
    [3],
    [4],
    [5],
    [9],
    [11],
    [22],
    [47],
    [101]
])
const target = matrix([
    [1],
    [1],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [1]
])

// create a prime guessing network
let nn = new Network(
    1, 30, 1,
    1_000_000,
    sigmoid,
    .2,
    10_000
)

nn.train(input, target)

nn.predict(input).forEach((v, i) => console.log(`output ${i}: `, v))

console.log()