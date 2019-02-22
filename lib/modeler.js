const Firestore = require('@google-cloud/firestore')
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')
const config = require(__dirname + '/config.js')
const math = require('mathjs')
const storage = require(__dirname + '/storage.js')

//const snapshot = require('../snapshot/export.js')

// Local globals
let isTraining = false
let step = 0
let losshistory = [];
let currloss;
let losshistory_history = [];
let epoch = 0
let model;

// Firestore connection
const db = new Firestore(config.firestore.init)

// File handling
const { promisify } = require('util')
const fs = require('fs')
const readFile = promisify(fs.readFile)
const writeFile = promisify(fs.writeFile)

async function calculateStatistics(snapshot) {
  let numbers = {}
  let totals = new Array(config.firestore.sort_order.length).fill(0)
  let lengths = new Array(config.firestore.sort_order.length).fill(0)
  let stats = {}

  let emoStats = {}
  
  snapshot.forEach(doc => {
    const data = doc.data()
    for(let i = 0; i < config.firestore.sort_order.length; i++) {
      if(!emoStats[data.title]) {
        emoStats[data.title] = {
          numbers: {},
          totals: new Array(config.firestore.sort_order.length).fill(0),
          lengths: new Array(config.firestore.sort_order.length).fill(0)
        }
      }
      let singledata = data.data[config.firestore.sort_order[i]]
      if(singledata != null) {
        emoStats[data.title].totals[i] += singledata
        emoStats[data.title].lengths[i] += 1
        if(!emoStats[data.title].numbers[i]) {
          emoStats[data.title].numbers[i] = [singledata]
        } else {
          emoStats[data.title].numbers[i].push(singledata)
        }
      }
    }
  })
  
  for (let key in emoStats) {
    if(!stats[key]) {
      stats[key] = {
        labels: config.firestore.sort_order,
        averages: [],
        stds: [],
        bounds: []
      }
    }
    stats[key].averages = emoStats[key].totals.map((n, i) => {
      return n/emoStats[key].lengths[i]
    })
    for(let i = 0; i < Object.keys(emoStats[key].numbers).length; i++) {
      let std = math.std(emoStats[key].numbers[i])
      let median = math.median(emoStats[key].numbers[i])
      stats[key].stds[i] = std
      stats[key].bounds[i] = {'upper': median + (std * config.stats.bound_multiplier), 'lower': median - (std * config.stats.bound_multiplier)}
    }
  }
  return stats
}

async function update() {
  // Simple sequential model
  model = await createModel()
  const results = db.collection(config.firestore.collection)
  const query = results.get()
  .then(async (snapshot) => {
    let dataset = []
    let stats = await calculateStatistics(snapshot)
    snapshot.forEach(doc => {
      let data = handleData(doc, stats)
      if(data) {
        dataset.push(data)
      }
    })
    tf.util.shuffle(dataset)
    return trainModel(dataset)
  })
  .catch(err => {
    console.log('Error getting documents', err)
    return false
  })
}

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
    batches.push({
      x: tf.tensor2d(xData, xShape),
      y: tf.tensor2d(yData, yShape)
    });

    index += batchSize
  }
  return runTraining(batches)
}

function handleData(data, stats) {
  let rawData = data.data()
  
  // Only return completed stuff
  if(rawData.completed) {
    
    // Take a copy of defaults
    let inputData = JSON.parse(JSON.stringify(config.tensorflow.emotion_defaults))

    // Make sure the output data is in required order
    outputData = []
    for (let i = 0; i < 12; i++) { // exclude shapes
      let key = config.firestore.sort_order[i]

      // Set to bound if over, clamping data
      if(rawData.data[key] > stats[rawData.title].bounds[i].upper) {
        rawData.data[key] = stats[rawData.title].bounds[i].upper
      } else if (rawData.data[key] < stats[rawData.title].bounds[i].lower) {
        rawData.data[key] = stats[rawData.title].bounds[i].lower
      }
      outputData.push(rawData.data[key])
    }

    if(config.tensorflow.include_shapes) {
      let shapeData = JSON.parse(JSON.stringify(config.tensorflow.shape_defaults))
      if(rawData.shape) {
        shapeData[rawData.shape.toLowerCase()] = 1
      }
      outputData = outputData.concat(Object.values(shapeData))
    }
    
    // Set value for selected emotion
    inputData[rawData.title.toLowerCase()] = 1

    outputObject = {
      x: Object.values(inputData),
      y: Object.values(outputData)
    }
    return outputObject
    
  } else {
    return false
  }
}

async function createModel() {
  const optimizer = tf.train.adamax(config.tensorflow.learning_rate);
  let model = tf.sequential()
  model.add(tf.layers.dense({units: 144, activation: 'linear', inputShape: [config.tensorflow.input_size]}))
  model.add(tf.layers.leakyReLU())
  model.add(tf.layers.dropout(0.25))
  model.add(tf.layers.dense({units: 72, activation: 'linear'}))
  model.add(tf.layers.leakyReLU())
  model.add(tf.layers.dropout(0.25))
  model.add(tf.layers.dense({units: 36, activation: 'linear'}))
  model.add(tf.layers.leakyReLU())
  model.add(tf.layers.dropout(0.25))
  model.add(tf.layers.dense({units: config.tensorflow.output_size, activation: 'relu'}))
  model.compile({optimizer: optimizer, loss: tf.losses.meanSquaredError, metrics: ['accuracy']})
  return model
}

// Trains and reports loss+accuracy for one batch of training data:
async function trainBatch(index, batches) {
  const history = await model.fit(batches[index].x, batches[index].y, {
    epochs: 1,
    shuffle: true,
    validationData: [batches[index].x, batches[index].y],
    batchSize: config.tensorflow.batch_size
  });

  step++;
  console.log('iteration: ', step)
  let loss = history.history.loss[0]
  let accuracy = history.history.acc[0];
  currloss = loss
  losshistory.push(loss)
  if(losshistory.length > config.tensorflow.loss_history) {
    losshistory.shift()
  }
  let avg = (losshistory.reduce(function(a, b) { return a + b; }, 0))/losshistory.length;
  console.log('accuracy: ', accuracy)
  console.log('averagedloss: ', avg)
  if(step%config.tensorflow.snapshot_interval === 0) {
    predict()
  }
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

async function runTraining(batches) {
  while (epoch < config.tensorflow.totalEpochs) {
    let train = await runEpochTrainAndVisual(batches)
    if(train === false) {
      break;
    }
    console.log('epoch', epoch)
    epoch++
  }
  let version = await readFile('./version.tag')
  await model.save('file://model/' + config.tensorflow.model_name + '-' + version)
  storage.upload('model/' + config.tensorflow.model_name + '-' + version + '/model.json')
  storage.upload('model/' + config.tensorflow.model_name + '-' + version + '/weights.bin')
  predict()
  await timeout(5500) // Timeout over export timeout, so version numbers stay in check in asynchonicity
  await writeFile('./version.tag', parseInt(version) + 1, 'utf8')
  storage.upload('./version.tag')
  return model
}

async function predict() {
  // console.log(config.firestore.sort_order)
  // const predictionOnes = await model.predictOnBatch(tf.ones([1, config.tensorflow.input_size]))
  // predictionOnes.print()
  // const predictionZeros = await model.predictOnBatch(tf.zeros([1, config.tensorflow.input_size]))
  // predictionZeros.print()
  const predictionAnger = await model.predictOnBatch(tf.tensor2d([1,0,0,0,0,0,0,0,0], [1,9]))
  const predictionSadness = await model.predictOnBatch(tf.tensor2d([0,0,0,1,0,0,0,0,0], [1,9]))
  const predictionSadAngry = await model.predictOnBatch(tf.tensor2d([1,0,0,1,0,0,0,0,0], [1,9]))
  predictionAnger.print()
  print(predictionAnger, 'Anger')
  print(predictionSadness, 'Sadness')
  print(predictionSadAngry, '😭 & 😡')
}

async function print(prediction, title){
  let values = prediction.dataSync()
  let valuesObject = {}
  for(let i = 0; i <config.firestore.sort_order.length; i++) {
    valuesObject[config.firestore.sort_order[i]] = values[i]
  }

  if(config.tensorflow.include_shapes) {
    let shapePredictions = values.slice(-10)
    let i = shapePredictions.indexOf(Math.max(...shapePredictions))
    valuesObject['shape'] = {
      'name' : Object.keys(config.tensorflow.shape_defaults)[i],
      'val': shapePredictions[i]
    }
  } else {
    valuesObject['shape'] = 'No shapes in model'
  }

  valuesObject['model'] = config.tensorflow.model_name
  valuesObject['iteration'] = step
  valuesObject['title'] = title
  valuesObject['loss'] = currloss
  //await snapshot.takesnap(valuesObject) // no snapshots for prod tests
}

async function timeout(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = {
  update: update,
  config: config
}