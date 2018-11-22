const config = {
  firestore: {
    init: {
      projectId: 'shape-mapper',
      keyFilename: 'credentials.json',
      timestampsInSnapshots: true
    },
    collection: 'results',
    sort_order: [ 
      'sliderValueRotZ',
      'sliderValueRotX',
      'sliderValueRotY',
      'sliderValueWidth',
      'sliderValueLength',
      'sliderValueHeight',
      'sliderValueScale',
      'sliderValueRadius',
      'sliderValueHue',
      'sliderValueLightness',
      'sliderValueOpacity',
      'sliderValueMatte',
    ]
  },
  tensorflow : {
    batch_size: 50,
    input_size: 9,
    output_size: 12,
    totalEpochs: 500000,
    snapshot_interval: 5000,
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
    },
    model_name: 'test-shaper-v2'
  }
}

module.exports = config