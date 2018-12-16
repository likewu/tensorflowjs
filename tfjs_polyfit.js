//import os
//os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2';

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const Util = require('util');
const FS = require('fs');
const PG = require('pg');
//const Redis = require("redis");
const Child_process = require('child_process');
const Async = require('async');
const Mathjs = require('mathjs');

Mathjs.import(require('numbers'), {wrap: true, silent: true});
Mathjs.import(require('numeric'), {wrap: true, silent: true});

var mService = {
  Database: null,
  Redis_cli5: null,
  Outdir: 'C:/Users/Administrator/Desktop/charles/',
  Comparedir: 'C:/Users/Administrator/Desktop/charles/report',
  Config: null
};
//mService.Config = require('./config.json');

function StartDatabase(callback) {
  var url = Util.format("postgres://%s:%s@%s:%d/%s",
    mService.Config.Database.User,
    mService.Config.Database.Password,
    mService.Config.Database.IP,
    mService.Config.Database.Port,
    mService.Config.Database.DB);
  console.log("Connect: " + url);
  PG.connect(url, function(err, db, done) {
    if (err) {
      console.log('Fail to connect database:' + err);
      throw err;
    }

    mService.Database = db;

    console.log('Connect to database sucess!!!');
    callback(err, db);
    });
}


const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));
//a.print();
//console.log(a);

/*const trueCoefficients = {a: -0.8, b: -0.2, c: 0.9, d: 0.5};
// 生成有误差的训练数据
const trainingData = generateData(7, trueCoefficients);
//console.log(trainingData.xs);
//console.log(trainingData.ys);
let xs = trainingData.xs;
let ys = trainingData.ys;*/
//process.exit();

let userInput = [
{x:1, y:2.274676}
,{x:2, y:2.264952}
,{x:3, y:2.302163}
,{x:4, y:2.292381}
,{x:5, y:2.313434}
,{x:6, y:2.352667}
,{x:7, y:2.461205}
,{x:8, y:2.780829}
,{x:9, y:2.974574}
,{x:10, y:2.977944}
];
let xs = [];
let ys = [];
for(let nn of userInput) {
	xs.push(nn.x);
	ys.push(nn.y);
}
xs = tf.tensor1d(xs);
ys = tf.tensor1d(ys);
let old_xs = xs;
let xmin = xs.min();
let xmax = xs.max();
let xrange = xmax.sub(xmin);
xs = xs.sub(xmin).div(xrange);
//xs = xs.div(xmax);
let ymin = ys.min();
let ymax = ys.max();
let yrange = ymax.sub(ymin);
ys = ys.sub(ymin).div(yrange);
//ys = ys.div(ymax);

xs.print();
ys.print();
train(xs, ys);
console.log('f(x) = '+a.dataSync()[0]+' * x ^ 3 + '+b.dataSync()[0]+' * x ^ 2 + '+c.dataSync()[0]+' * x + '+d.dataSync()[0]);
/*a.print();
b.print();
c.print();
d.print();*/

// 预测数据
const predictionsAfter = predict(xs);
old_xs.print();
//yrange.print();
//predictionsAfter.print();
predictionsAfter.mul(yrange).add(ymin).print();
	
function predict(x) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(3))) // a * x^3
      .add(b.mul(x.square())) // + b * x ^ 2
      .add(c.mul(x)) // + c * x
      .add(d); // + d
  });
}

function loss(predictions, labels) {
  // 将labels（实际的值）进行抽象
  // 然后获取平均数.
  let meanSquareError = predictions.sub(labels).square().mean();
  return meanSquareError;
}

function train(xs, ys, numIterations = 75) {
  let learningRate = 0.5;
  let optimizer = tf.train.sgd(learningRate);

  for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => {
      let predsYs = predict(xs);
      //predsYs.print();
      let step = loss(predsYs, ys);
      //step.print();
      return step;
    });
  }
}

function generateData(numPoints, coeff, sigma = 0.04) {
  return tf.tidy(() => {
    const [a, b, c, d] = [
      tf.scalar(coeff.a),
      tf.scalar(coeff.b),
      tf.scalar(coeff.c),
      tf.scalar(coeff.d)
    ];
  
    const xs = tf.randomUniform([numPoints], -1, 1);
    const ys = a.mul(xs.pow(tf.scalar(3, 'int32')))
      .add(b.mul(xs.square()))
      .add(c.mul(xs))
      .add(d)
      .add(tf.randomNormal([numPoints], 0, sigma));
    //console.log(ys);
    const ymin = ys.min();
    const ymax = ys.max();
    const yrange = ymax.sub(ymin);
    const ysNormalized = ys.sub(ymin).div(yrange);
 
    return {
      xs,
      ys: ysNormalized
    };
  })
}
