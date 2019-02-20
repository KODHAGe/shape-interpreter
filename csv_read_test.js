// Based on this nice observable by Nick Kreeger https://beta.observablehq.com/@nkreeger/visualizing-ml-training-using-tensorflow-js-and-baseball-d
// Proof of concept reading data f  rom csv
const tf = require('@tensorflow/tfjs')
const csv = require('csvtojson')

const config = {
  batch_size: 50,
  input_size: 9,
  output_size: 10
}

// Load the binding:
require('@tensorflow/tfjs-node')

// Train a simple model:
const model = tf.sequential()
model.add(tf.layers.dense({units: 64, activation: 'relu', inputShape: [config.input_size]}))
model.add(tf.layers.dropout(0.5))
model.add(tf.layers.dense({units: 64, activation: 'relu'}))
model.add(tf.layers.dropout(0.5))
model.add(tf.layers.dense({units: config.output_size, activation: 'linear'}))
model.compile({optimizer: tf.train.adam(0.01), loss: 'categoricalCrossentropy'})

const batch_size = config.batch_size;

// CSV
const csvFilePath='data/testdata.csv'
csv().fromFile(csvFilePath).then((json) => {
  let data = []
  json.forEach((row) => {
    let x = []
    let y = []
    let keys = Object.keys(row)
    keys.forEach((key, index) => {
      if(index < config.input_size) {
        x.push(parseFloat(row[key]))
      }
      if(index >= config.input_size) {
        y.push(parseFloat(row[key]))
      }
    })
    data.push({x: x, y: y})
  })
  console.log(data)
  tf.util.shuffle(data)

  const batches = []
  let index = 0
  let batch_size = batch_size
  while (index < data.length) {
    if (data.length - index < batch_size) {
      batch_size = data.length - index
    }

    const dataBatch = data.slice(index, index + batch_size);
    const xShape = [dataBatch.length, config.input_size]
    const yShape = [dataBatch.length, config.output_size]
    const xData = new Float32Array(tf.util.sizeFromShape(xShape))
    const yData = new Float32Array(tf.util.sizeFromShape(yShape))

    let xoffset = 0
    let yoffset = 0
    for (let i = 0; i < dataBatch.length; i++) {
      const xyData = dataBatch[i]

      let x = []
      for(let n = 0; n < config.input_size; n++) {
        x.push(xyData.x[n])
      }
      xData.set(x, xoffset)
      xoffset += config.input_size;
    }
    for (let i = 0; i < dataBatch.length; i++) {
      const xyData = dataBatch[i]

      let y = []
      for(let n = 0; n < config.output_size; n++) {
        y.push(xyData.y[n])
      }
      yData.set(y, yoffset)

      yoffset += config.output_size
    }

    // Make tensors, push to batches:
    batches.push({
      x: tf.tensor2d(xData, xShape),
      y: tf.tensor2d(yData, yShape)
    });

    index += batch_size
  }
  runTraining(batches)
})

let isTraining = false
let step = 0
let losshistory = [0,0,0,0,0,0,0,0,0,0]
// Trains and reports loss+accuracy for one batch of training data:
async function trainBatch(index, batches) {
  const history = await model.fit(batches[index].x, batches[index].y, {
    epochs: 1,
    shuffle: false,
    validationData: [batches[index].x, batches[index].y],
    batch_size: batch_size
  });

  step++;
  console.log('iteration: ',step)
  let loss = history.history.loss[0]
  losshistory.push(loss)
  let avg = (losshistory.reduce(function(a, b) { return a + b; }, 0))/losshistory.length;
  console.log('averagedloss: ', avg)

  await tf.nextFrame()
  await tf.nextFrame()
}

async function runEpochTrainAndVisual(batches) {
  isTraining = true;

  for (let i = 0; i < batches.length; i++) {
    await trainBatch(i, batches)
    await tf.nextFrame()
    await tf.nextFrame()
  }
  isTraining = false
}

let epoch = 0
let totalEpochs = 200
async function runTraining(batches) {
  while (epoch < totalEpochs) {
    await runEpochTrainAndVisual(batches)
    epoch++
  }
  await model.save('file://model/test-model' + new Date().getTime() + '.json')
  predict()
}

function predict() {
  const predictions = model.predictOnBatch(tf.ones([1, config.input_size]))
  predictions.print()
  const predictions2 = model.predictOnBatch(tf.zeros([1, config.input_size]))
  predictions2.print()
}
