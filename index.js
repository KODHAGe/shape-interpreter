const tf = require('@tensorflow/tfjs')
const csv = require('csvtojson')

// Load the binding:
require('@tensorflow/tfjs-node')

// Train a simple model:
const model = tf.sequential()
model.add(tf.layers.dense({units: 24, activation: 'relu', inputShape: [12]}))
model.add(tf.layers.dense({units: 9, activation: 'linear'}))
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'})

const batch_size = 50;

// CSV
const csvFilePath='data/test.csv'
csv().fromFile(csvFilePath).then((json) => {
  const data = []
  json.forEach((value) => {
    const x = []
    x.push(parseFloat(value.dummy1))
    x.push(parseFloat(value.dummy2))
    x.push(parseFloat(value.dummy3))
    x.push(parseFloat(value.dummy4))
    x.push(parseFloat(value.dummy5))
    x.push(parseFloat(value.dummy6))
    x.push(parseFloat(value.dummy7))
    x.push(parseFloat(value.dummy8))
    x.push(parseFloat(value.dummy9))
    x.push(parseFloat(value.dummy10))
    x.push(parseFloat(value.dummy11))
    x.push(parseFloat(value.dummy12))
    const y = []
    y.push(parseFloat(value.x))
    y.push(parseFloat(value.y))
    y.push(parseFloat(value.z))
    y.push(parseFloat(value.uno))
    y.push(parseFloat(value.dos))
    y.push(parseFloat(value.tres))
    y.push(parseFloat(value.h))
    y.push(parseFloat(value.s))
    y.push(parseFloat(value.v))
    data.push({x: x, y: y})
  })
  
  tf.util.shuffle(data)

  const batches = [];
  let index = 0;
  let batchSize = batch_size;
  while (index < data.length) {
    if (data.length - index < batch_size) {
      batchSize = data.length - index;
    }

    const dataBatch = data.slice(index, index + batchSize);
    const shape = [dataBatch.length, 12]
    const yShape = [dataBatch.length, 9]
    const xData = new Float32Array(tf.util.sizeFromShape(shape))
    const yData = new Float32Array(tf.util.sizeFromShape(yShape))

    let xoffset = 0
    for (let i = 0; i < dataBatch.length; i++) {
      const xyData = dataBatch[i]

      const x = [];
      x.push(xyData.x[0])
      x.push(xyData.x[1])
      x.push(xyData.x[2])
      x.push(xyData.x[3])
      x.push(xyData.x[4])
      x.push(xyData.x[5])
      x.push(xyData.x[6])
      x.push(xyData.x[7])
      x.push(xyData.x[8])
      x.push(xyData.x[9])
      x.push(xyData.x[10])
      x.push(xyData.x[11])
      xData.set(x, xoffset)
      
      xoffset += 12;
    }
    let yoffset = 0
    for (let i = 0; i < dataBatch.length; i++) {
      const xyData = dataBatch[i]
      const y = []
      y.push(xyData.y[4])
      y.push(xyData.y[1])
      y.push(xyData.y[2])
      y.push(xyData.y[3])
      y.push(xyData.y[4])
      y.push(xyData.y[5])
      y.push(xyData.y[6])
      y.push(xyData.y[7])
      y.push(xyData.y[8])
      yData.set(y, yoffset)

      yoffset += 9;
    }

    // Push batch tensor:
    batches.push({
      x: tf.tensor2d(xData, shape),
      y: tf.tensor2d(yData, yShape)
    });

    index += batchSize;
  }
  runTraining(batches)
})

let isTraining = false 
let step = 0

// Trains and reports loss+accuracy for one batch of training data:
async function trainBatch(index, batches) {
  const history = await model.fit(batches[index].x, batches[index].y, {
    epochs: 1,
    shuffle: false,
    validationData: [batches[index].x, batches[index].y],
    batchSize: batch_size
  });
      
  step++;
  console.log(step)
  let loss = history.history.loss[0];
  //let accuracy = history.history.acc[0];
  console.log('loss ',loss)
    
  await tf.nextFrame();
  //updateHeatmap();
  await tf.nextFrame();
}

async function runEpochTrainAndVisual(batches) {
  isTraining = true;
  
  for (let i = 0; i < batches.length; i++) {
    await trainBatch(i, batches);
    
    // The tf.nextFrame() helper function returns a Promise when requestAnimationFrame()
    // has completed. These calls ensure that the single-threaded JS event loop is not
    // blocked during training:
    await tf.nextFrame();

    //updateHeatmap();
    await tf.nextFrame();
  }
  
  isTraining = false;
}

// Function that ensures the model is trained to the number of epochs selected
// by the user in this codelab:
let epoch = 0
let totalEpochs = 1
async function runTraining(batches) {
  while (epoch < totalEpochs) {
    await runEpochTrainAndVisual(batches)
    epoch++
  }
  await model.save('file://model/');
  predict()
}

function predict() {
  const predictions = model.predictOnBatch(tf.ones([1, 12]))
  predictions.print()
}