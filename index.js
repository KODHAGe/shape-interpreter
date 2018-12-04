require('dotenv').config()
const path = require('path')
const jwt = require('jwt-simple')
const secret = process.env.JWT_SECRET

// Service/server
const { send, json } = require('micro')
const { router, get, post, options } = require('microrouter')
const handler = require('serve-handler');

// File handling
const { promisify } = require('util')
const fs = require('fs')
const readdir = promisify(fs.readdir)

// Tensorflow
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')
const modeler = require('./lib/modeler.js')

async function get_latest_model() {
  try {
    files = await readdir('model/')
  }
  finally {
    files.sort(function(a, b) {
      a = path.join(__dirname, '/model/', a)
      a = fs.statSync(a).ctime
      b = path.join(__dirname, '/model/', b)
      b = fs.statSync(b).ctime
      return a - b
    })
    files.splice(files.indexOf('.DS_Store'), 1);
    let latest = files[files.length - 1]
    console.log(latest)
    return latest
  }
}

const authenticate = async (req, res, callback) => {
  let body = await json(req)
  let token = body.jwt
  try {
    let decoded = jwt.decode(token, secret)
    if (decoded) {
      res.setHeader('Access-Control-Allow-Origin', '*')
      // Add specific authentication scope handling here
      return callback(req, res)
    }
  } catch (err) {
    send(res, 401, 'Authentication failed: ' + err.message)
    return false
  }
}

async function update_model(req, res) {
  await authenticate(req, res, async() => {
    modeler.update()
    let latest = get_latest_model()
    send(res, 200, 'Training the model. This will take some time. Current version is '+ latest + '.')  
  })
}

async function get_model(req, res) {
  await authenticate(req, res, async() => {
    let latest = await get_latest_model()
    const model = await tf.loadModel('file://model/' + latest + '/model.json')
    send(res, 200, model)
  })
}

async function make_prediction (req, res) {
  await authenticate(req, res, async() => {
    // default to zeroes
    let array;
    let body = await json(req)
    if(!body.array) {
      array = [0,0,0,0,0,0,0,0,0]
    } else if (body.array.length != 9) {
      send(res, 500, 'Array of invalid size.')
    } else {
      array = body.array
    }
    let latest = await get_latest_model()
    const model = await tf.loadModel('file://model/' + latest + '/model.json')
    let prediction = await model.predictOnBatch(tf.tensor2d(array, [1,9]))
    let values = prediction.dataSync()
    let order = modeler.config.firestore.sort_order
    console.log(order)
    console.log(values)
    let predictionResultObject = {}
    for (let i = 0; i < order.length; i++) {
      predictionResultObject[order[i]] = values[i]
    }
    predictionResultObject['modelVersion'] = latest
    send(res, 200, predictionResultObject)
  })
}

async function handle_preflight (req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
  res.setHeader('Access-Control-Allow-Credentials', false)
  res.setHeader('Access-Control-Max-Age', '86400')
  res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With, X-HTTP-Method-Override, Content-Type, Accept')
  send(res, 200)
}

async function handle (req, res) {
  return handler(req, res, {
    "rewrites": [
      { "source": "./index", "destination": "/snapshot/output" },
    ]
  })
}

module.exports = router(
  get('/', (req, res) => {
    send(res, 200, 'Shape interpreter API v1')
  }),
  get('*', handle),
  options('*', handle_preflight),
  post('/updateModel', update_model),
  post('/getModel', get_model),
  post('/makePrediction', make_prediction)
)