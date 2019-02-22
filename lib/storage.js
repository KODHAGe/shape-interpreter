const fs = require('fs')
const config = require(__dirname + '/config.js')
const {Storage} = require('@google-cloud/storage')

// Storage
const storage = new Storage(config.storage.init)
const bucket = storage.bucket(config.storage.bucket)

async function upload(file) {
  await bucket.upload('./' + file, { 'destination': file })
}

async function download(file, path = __dirname + '/') {
  // Create the folder first because we live in the dark ages, don't we?

  if (!fs.existsSync(path)){
    fs.mkdirSync(path, { recursive: true });
  }
  await bucket.file(file).download({'destination': './' + file})
}

module.exports = {
  upload: upload,
  download: download
}