const Firestore = require('@google-cloud/firestore')
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

const config = {
  firestore: {
    init: {
      projectId: 'shape-mapper',
      keyFilename: 'credentials.json',
      timestampsInSnapshots: true
    },
    collection: 'results'
  },
  tensorflow : {
    batch_size: 5,
    input_size: 9,
    output_size: 12,
    emotion_defaults : {
      'anger' : 0, 
      'fear' : 0, 
      'joy' : 0, 
      'sadness' : 0, 
      'analytical' : 0, 
      'confident' : 0, 
      'tentative' : 0, 
      'negative' : 0, 
      'positive' : 0
    }
  }
}

// Firestore connection
const db = new Firestore(config.firestore.init)

// Simple sequential model
const model = createModel()

const results = db.collection(config.firestore.collection)
var query = results.get()
  .then(snapshot => {
    var dataset = []
    snapshot.forEach(doc => {
      let data = handleData(doc)
      if(data) {
        dataset.push(
          {
            x: data.x,
            y: data.y
          }
        )
      }
    })
    console.log(dataset)
    tf.util.shuffle(dataset)
    trainModel(dataset)
  })
  .catch(err => {
    console.log('Error getting documents', err);
  })

function trainModel(data) {
  const batches = []
  let index = 0
  let batchSize = config.tensorflow.batch_size
  while (index < data.length) {
    if (data.length - index < batchSize) {
      batchSize = data.length - index
    }
    const dataBatch = data.slice(index, index + batchSize);
    const xShape = [dataBatch.length, config.tensorflow.input_size]
    const yShape = [dataBatch.length, config.tensorflow.output_size]
    const xData = new Float32Array(tf.util.sizeFromShape(xShape))
    const yData = new Float32Array(tf.util.sizeFromShape(yShape))

    let xoffset = 0
    let yoffset = 0
    for (let i = 0; i < dataBatch.length; i++) {
      const xyData = dataBatch[i]

      let x = []
      for(let n = 0; n < config.tensorflow.input_size; n++) {
        x.push(xyData.x[n])
      }
      xData.set(x, xoffset)
      xoffset += config.tensorflow.input_size;
    }
    for (let i = 0; i < dataBatch.length; i++) {
      const xyData = dataBatch[i]

      let y = []
      for(let n = 0; n < config.tensorflow.output_size; n++) {
        y.push(xyData.y[n])
      }
      yData.set(y, yoffset)

      yoffset += config.tensorflow.output_size
    }

    // Make tensors, push to batches:
    console.log(xData)
    console.log(xShape)
    batches.push({
      x: tf.tensor2d(xData, xShape),
      y: tf.tensor2d(yData, yShape)
    });

    index += batchSize
  }
  runTraining(batches)
}

function handleData(data) {
  let rawData = data.data()
  // Only return completed stuff
  if(rawData.completed) {
    // Make it flat
    let inputData = config.tensorflow.emotion_defaults
    let outputData = rawData.data
    // Set value for selected emotion
    inputData[rawData.title.toLowerCase()] = 1
    outputObject = {
      x_headers: Object.keys(inputData),
      y_headers: Object.keys(outputData),
      x: Object.values(inputData),
      y: Object.values(outputData)
    }
    return outputObject
  } else {
    return false
  }
}

function createModel() {
  let model = tf.sequential()
  model.add(tf.layers.dense({units: 64, activation: 'relu', inputShape: [config.tensorflow.input_size]}))
  model.add(tf.layers.dropout(0.5))
  model.add(tf.layers.dense({units: 64, activation: 'relu'}))
  model.add(tf.layers.dropout(0.5))
  model.add(tf.layers.dense({units: config.tensorflow.output_size, activation: 'linear'}))
  model.compile({optimizer: 'adamax', loss: 'meanSquaredError'})
  return model
}

let isTraining = false
let step = 0
let losshistory = [0,0,0,0,0,0,0,0,0,0]
// Trains and reports loss+accuracy for one batch of training data:
async function trainBatch(index, batches) {
  const history = await model.fit(batches[index].x, batches[index].y, {
    epochs: 1,
    shuffle: false,
    validationData: [batches[index].x, batches[index].y],
    batchSize: config.tensorflow.batch_size
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
let totalEpochs = 500
async function runTraining(batches) {
  while (epoch < totalEpochs) {
    await runEpochTrainAndVisual(batches)
    epoch++
  }
  await model.save('file://model/test-model' + new Date().getTime() + '.json')
  predict()
}

function predict() {
  const predictions = model.predictOnBatch(tf.ones([1, config.tensorflow.input_size]))
  predictions.print()
  const predictions2 = model.predictOnBatch(tf.zeros([1, config.tensorflow.input_size]))
  predictions2.print()
  const predictionsAnger = model.predictOnBatch(tf.oneHot(tf.tensor1d([0, 1], 'int32'), config.tensorflow.input_size))
  predictionsAnger.print()
}