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
      'box',
      'cone',
      'cylinder',
      'dodecahedron',
      'ellipsoid',
      'plane',
      'icosahedron',
      'torus',
      'octahedron',
      'tetrahedron'
    ]
  },
  stats: {
    bound_multiplier: 1
  },
  tensorflow : {
    batch_size: 5,
    input_size: 9,
    output_size: 22, // 12 without shapes
    include_shapes: true,
    totalEpochs: 3000,
    snapshot_interval: 1000,
    learning_rate: 0.002,
    loss_history: 100,
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
    shape_defaults : {
      'box' : 0, 
      'cone' : 0, 
      'cylinder' : 0, 
      'dodecahedron' : 0, 
      'ellipsoid' : 0, 
      'plane' : 0, 
      'icosahedron' : 0, 
      'torus' : 0, 
      'octahedron' : 0,
      'tetrahedron' : 0
    },
    model_name: 'shaper'
  }
}

module.exports = config
