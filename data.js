const tf = require('@tensorflow/tfjs');
const Canvas = require('canvas');
const echarts = require('echarts');
const fetch = require("fetch");
const fs = require("fs");
const getPixels = require("get-pixels")

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const MNIST_IMAGES_SPRITE_PATH = 'temp/resource/mnist_images.png';
const MNIST_LABELS_PATH = 'temp/resource/mnist_labels_uint8';

class MnistData {
  constructor(TRAIN_TEST_RATIO) {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
    this.NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
    this.NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - this.NUM_TRAIN_ELEMENTS;
  }

  async load() {
    let that = this;
    this.trainIndices_idx1 = [];

    const labelsRequest = new Promise((resolve, reject) => {
      const fetchreq = fs.readFile(MNIST_LABELS_PATH, (err, chunk) => {
          that.datasetLabels = chunk.slice(0, NUM_CLASSES * NUM_DATASET_ELEMENTS);
          //console.log(that.datasetLabels.length);
          //console.log(that.datasetLabels[0],that.datasetLabels[1],that.datasetLabels[2],that.datasetLabels[3],that.datasetLabels[4]
          //  ,that.datasetLabels[5],that.datasetLabels[6],that.datasetLabels[7],that.datasetLabels[8],that.datasetLabels[9]);
          resolve();
      });
    });
    await labelsRequest;

    const imgRequest = new Promise((resolve, reject) => {
      getPixels(MNIST_IMAGES_SPRITE_PATH, function(err, pixels) {
          if(err) {
            console.log("Bad image path")
            return
          }
          //console.log("got pixels", pixels.shape.slice())
          //console.log("got pixels", pixels.data)
          const imgwidth = pixels.shape[0];
          const chunkSize = 5000;
          let imgrow = 0;
          for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
            const datasetBytesView = new Float32Array(
                pixels.data.buffer, i * IMAGE_SIZE * chunkSize * 4,
                IMAGE_SIZE * chunkSize);

            const imageData = new Uint8Array(pixels.data.buffer, i * IMAGE_SIZE * chunkSize * 4, IMAGE_SIZE * chunkSize * 4);

            for (let j = 0; j < imageData.length / 4; j++) {
              // All channels hold an equal value since the image is grayscale, so
              // just read the red channel.
              datasetBytesView[j] = imageData[j * 4] / 255;
              if((j+1) % IMAGE_SIZE == 0) {
                let label = that.datasetLabels.slice(NUM_CLASSES * imgrow, NUM_CLASSES * (imgrow+1));
                if(label[7]==1) that.trainIndices_idx1.push(imgrow);
                /*if(imgrow<50||imgrow>=64950) {
                  if(label[1]==1) {
                    let labelstr = 99;
                    for (let k = 0; k < label.length; k++) {
                      if(label[k]==1) {labelstr = k; break;}
                    }
                    const canvas1 = Canvas.createCanvas(28,28);
                    const mnist1 = datasetBytesView.slice(j+1-IMAGE_SIZE, j+1);
                    draw(mnist1, canvas1, 'mnist'+imgrow+'-'+labelstr+'.png');
                  }
                }*/
                imgrow++;
              }
            }
            /*if(i==0) {
              const canvas11 = Canvas.createCanvas(784,5000);
              draw(datasetBytesView, canvas11, 'mnist11.png');
            }*/
          }
          that.datasetImages = new Float32Array(pixels.data.buffer);
          resolve();
      });
    });
    await imgRequest;
    console.log(this.trainIndices_idx1.length);

    /*const [imgResponse, labelsResponse] =
        await Promise.all([imgRequest, labelsRequest]);*/

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(this.NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(this.NUM_TEST_ELEMENTS);
    //console.log('trainIndices[1] ',this.trainIndices[1]);
    //console.log('trainIndices[2] ',this.trainIndices[2]);

    // Slice the the images and labels into train and test sets.
    this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * this.NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * this.NUM_TRAIN_ELEMENTS);

    this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * this.NUM_TRAIN_ELEMENTS);
    this.testLabels = this.datasetLabels.slice(NUM_CLASSES * this.NUM_TRAIN_ELEMENTS);
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
        batchSize, [this.trainImages, this.trainLabels], () => {
          let cur_idx = this.shuffledTrainIndex;

          /*while(true) {
            if(this.trainIndices_idx1.includes(cur_idx)) break;
            cur_idx++;
          }*/

          this.shuffledTrainIndex = (cur_idx + 1) % this.testIndices.length;
          //return cur_idx;
          return this.trainIndices[this.shuffledTrainIndex];
        });
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
          (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();
      //console.log('idx ',idx);

      const image =
          data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label =
          data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
      batchLabelsArray.set(label, i * NUM_CLASSES);

      /*let labelstr = 99;
      for (let k = 0; k < label.length; k++) {
        if(label[k]==1) {labelstr = k; break;}
      }
      const canvas1 = Canvas.createCanvas(28,28);
      draw(image, canvas1, 'b_mnist'+idx+'-'+labelstr+'.png');*/
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return {xs, labels};
  }


  async load2() {
    let that = this;
    /*
    const canvas = Canvas.createCanvas(784,1);
    const ctx = canvas.getContext('2d');

    const imgRequest = new Promise((resolve, reject) => {
      Canvas.loadImage(MNIST_IMAGES_SPRITE_PATH).then((img) => {
          const datasetBytesBuffer =
              new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

          const chunkSize = 5000;
          canvas.width = img.width;
          canvas.height = chunkSize;

          for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
            const datasetBytesView = new Float32Array(
                datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
                IMAGE_SIZE * chunkSize);
            ctx.drawImage(
                img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
                chunkSize);

            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

            for (let j = 0; j < imageData.data.length / 4; j++) {
              // All channels hold an equal value since the image is grayscale, so
              // just read the red channel.
              datasetBytesView[j] = imageData.data[j * 4] / 255;
            }
          }
          that.datasetImages = new Float32Array(datasetBytesBuffer);
          //const canvas1 = Canvas.createCanvas(28,28);
          //const mnist1 = that.datasetImages.slice(2 * IMAGE_SIZE, IMAGE_SIZE * 3);
          //draw(mnist1, canvas1, 'mnist3.png');
          resolve();
      });
    });*/
  }
}

function draw(image, canvas, filename) {
  const [width, height] = [canvas.width, canvas.height];
  //canvas.width = width;
  //canvas.height = height;
  const ctx = canvas.getContext("2d");
  const imageData = Canvas.createImageData(width, height);
  const data = image;//image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    //if(data[i]!=0) console.log(data[i]);
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

module.exports = MnistData;