//import * as tf from '@tensorflow/tfjs';
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const Canvas = require('canvas');
const fs = require('fs')

const MnistData = require('./data');
const GANLabModel = require('./ganlab_models');

// Input params
const BATCH_SIZE = 200;
const SIZE = 28;
const INPUT_SIZE = SIZE * SIZE;
const SEED_SIZE = 40;
const SEED_STD = 3.5;
const ONES = tf.ones([BATCH_SIZE, 1]);
const ONES_PRIME = tf.ones([BATCH_SIZE, 1]).mul(tf.scalar(1));
const ZEROS = tf.zeros([BATCH_SIZE, 1]);

// Generator and discrimantor params
const DISCRIMINATOR_LEARNING_RATE = 0.025;
const GENERATOR_LEARNING_RATE = 0.025;
const dOptimizer = tf.train.sgd(DISCRIMINATOR_LEARNING_RATE);
const gOptimizer = tf.train.sgd(GENERATOR_LEARNING_RATE);

// Helper functions
const varInitNormal = (shape, mean = 0, std = 0.1) =>
  tf.variable(tf.randomNormal(shape, mean, std));
const varLoad = (shape, data) => tf.variable(tf.tensor(shape, data));
const seed = (s = BATCH_SIZE) => tf.randomNormal([s, SEED_SIZE], 0, SEED_STD);

// Network arch for generator
let G1w = varInitNormal([SEED_SIZE, 140]);
let G1b = varInitNormal([140]);
let G2w = varInitNormal([140, 80]);
let G2b = varInitNormal([80]);
let G3w = varInitNormal([80, INPUT_SIZE]);
let G3b = varInitNormal([INPUT_SIZE]);

// Network arch for discriminator
let D1w = varInitNormal([INPUT_SIZE, 200]);
let D1b = varInitNormal([200]);
let D2w = varInitNormal([200, 90]);
let D2b = varInitNormal([90]);
let D2w1 = varInitNormal([90, 90]);
let D2b1 = varInitNormal([90]);
let D3w = varInitNormal([90, 1]);
let D3b = varInitNormal([1]);


let noiseSize = 2;
let numGeneratorLayers = 1;
let numDiscriminatorLayers = 1;
let numGeneratorNeurons = 1;
let numDiscriminatorNeurons = 1;
let lossTypeOptions = ['Log loss', 'LeastSq loss'];
let lossType = 'Log loss';
let model = new GANLabModel(
      noiseSize, numGeneratorLayers, numDiscriminatorLayers,
      numGeneratorNeurons, numDiscriminatorNeurons,
      BATCH_SIZE, lossType);
model.initializeModelVariables();
/*model.updateOptimizer('D', this.dOptimizerType, this.dLearningRate);
model.updateOptimizer('G', this.gOptimizerType, this.gLearningRate);
*/


function gen(xs) {
  const l1 = tf.leakyRelu(xs.matMul(G1w).add(G1b));
  const l2 = tf.leakyRelu(l1.matMul(G2w).add(G2b));
  const l3 = tf.tanh(l2.matMul(G3w).add(G3b));
  return l3;
}

function disReal(xs) {
  const l1 = tf.leakyRelu(xs.matMul(D1w).add(D1b));
  const l2 = tf.leakyRelu(l1.matMul(D2w).add(D2b));
  const logits = l2.matMul(D3w).add(D3b);
  const output = tf.sigmoid(logits);
  return [logits, output];
}

function disFake(xs) {
  return disReal(gen(xs));
}

// this method is remove from 13.0, but available in 12.0
function sigmoidCrossEntropyWithLogits(target, output) {
  return tf.tidy(function() {
    let maxOutput = tf.maximum(output, tf.zerosLike(output));
    let outputXTarget = tf.mul(output, target);
    let sigmoidOutput = tf.log(
      tf.add(tf.scalar(1.0), tf.exp(tf.neg(tf.abs(output))))
    );
    let result = tf.add(tf.sub(maxOutput, outputXTarget), sigmoidOutput);
    return result;
  });
}

async function trainBatch(realBatch, fakeBatch) {
  const dcost = dOptimizer.minimize(
    () => {
      const [logitsReal, outputReal] = disReal(realBatch);
      const [logitsFake, outputFake] = disFake(fakeBatch);

      const lossReal = sigmoidCrossEntropyWithLogits(ONES_PRIME, logitsReal);
      const lossFake = sigmoidCrossEntropyWithLogits(ZEROS, logitsFake);
      return lossReal.add(lossFake).mean();
    },
    true,
    [D1w, D1b, D2w, D2b, D3w, D3b]
  );
  await tf.nextFrame();
  const gcost = gOptimizer.minimize(
    () => {
      const [logitsFake, outputFake] = disFake(fakeBatch);

      const lossFake = sigmoidCrossEntropyWithLogits(ONES, logitsFake);
      return lossFake.mean();
    },
    true,
    [G1w, G1b, G2w, G2b, G3w, G3b]
  );
  await tf.nextFrame();

  return [dcost, gcost];
}

const TRAIN_TEST_RATIO = 64990 / 65000;
let data;
async function mnist() {
  data = new MnistData(TRAIN_TEST_RATIO);
  await data.load();
  console.log("Data loaded!");
}

mnist().then(async function() {
  //return;
  async function aa() { //trainBtn
    const TRAIN_BATCHES = 1200;
    for (let i = 0; i < TRAIN_BATCHES; i++) {
      const real = data.nextTrainBatch(BATCH_SIZE);
      const fake = seed();
      const [dcost, gcost] = await trainBatch(real.xs, fake);
      if(i%50==0)console.log(i);
      //update_chart(cost_chart, dcost, gcost, i);
    }
  }
  await aa();

  async function bb() {  //genBtn
    await tf.nextFrame();
    const data = gen(seed(1));
    const canvas = Canvas.createCanvas(28,28);
    draw(data, canvas, 'test.png');
  }
  await bb();
});

/*private async onClickSaveModelButton() {
  const dTensors: tf.NamedTensorMap = 
    this.model.dVariables.reduce((obj, item, i) => {
      obj[`d-${i}`] = item;
      return obj;
    }, {});
  const gTensors: tf.NamedTensorMap = 
    this.model.gVariables.reduce((obj, item, i) => {
      obj[`g-${i}`] = item;
      return obj;
    }, {});
  const tensors: tf.NamedTensorMap = {...dTensors, ...gTensors};

  const modelInfo: {} = {
    'shape_name': this.selectedShapeName,
    'iter_count': this.iterationCount,
    'config': {
      selectedNoiseType: this.selectedNoiseType,
      noiseSize: this.noiseSize,
      numGeneratorLayers: this.numGeneratorLayers,
      numDiscriminatorLayers: this.numDiscriminatorLayers,
      numGeneratorNeurons: this.numGeneratorNeurons,
      numDiscriminatorNeurons: this.numDiscriminatorNeurons,
      dLearningRate: this.dLearningRate,
      gLearningRate: this.gLearningRate,
      dOptimizerType: this.dOptimizerType,
      gOptimizerType: this.gOptimizerType,
      lossType: this.lossType,
      kDSteps: this.kDSteps,
      kGSteps: this.kGSteps,
    }
  };
  const weightDataAndSpecs = await tf.io.encodeWeights(tensors);
  const modelArtifacts: tf.io.ModelArtifacts = {
    modelTopology: modelInfo,
    weightSpecs: weightDataAndSpecs.specs,
    weightData: weightDataAndSpecs.data,
  };

  const downloadTrigger = 
    tf.io.getSaveHandlers('downloads://ganlab_trained_model')[0];
  await downloadTrigger.save(modelArtifacts);
}*/

function draw(image, canvas, filename) {
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const imageData = Canvas.createImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);

  const out = fs.createWriteStream(__dirname + '/temp/' + filename)
  const stream = canvas.createPNGStream()
  stream.pipe(out)
  out.on('finish', () =>  console.log('The PNG file was created.'))
}