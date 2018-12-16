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
//mService.Config = require('./config10.json');

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

////////////////////////////////////////////////////////////////
var inputMatrix = [];
Async.series([
    function(cb){                       //连接数据库
        //StartDatabase(cb);
        cb();
    },
    function(cb){
        /*mService.Redis_cli5 = Redis.createClient(mService.Config.Cache_redis_db5[1], mService.Config.Cache_redis_db5[0]);
        mService.Redis_cli5.select(mService.Config.Cache_redis_db5[2], function(err,value) {
            cb(err, mService.Redis_cli5);
        });
        console.log('redis connect secess!');*/
        cb();
    },
    function(cb){                      //读取文件
        let userInput = [
             {x:29, y:86}
            ,{x:70, y:19}
            ,{x:99, y:83}
            ,{x:57, y:94}
            ,{x:71, y:88}
            ,{x:96, y:50}
            ,{x:21, y:50}
        ];
        let ret = polyfit(userInput);
        console.log(ret);
        console.log(inputMatrix);

        cb();
    }
],function(err,result){
    if (err) throw err;
    console.log('complete');
});


/**
 * Helper function to output a value in the console. Value will be formatted.
 * @param {*} value
 */
function print (value) {
  var precision = 14;
  console.log(Mathjs.format(value, precision));
}

function polyfit(userInput) {
    let returnResult = [];
    inputMatrix = [];
    let n = userInput.length;
    for(let i = 0; i < n; i++) {
        let tempArr = [];
        for(let j = 0; j < n; j++) {
            tempArr.push(Math.pow(userInput[i].x, n - j - 1));
        }
        tempArr.push(userInput[i].y);
        inputMatrix.push(tempArr);
    }
    for(let i = 0; i < n; i++) {
        let base = inputMatrix[i][i];
        for(let j = 0; j < n + 1; j++) {
            if(base == 0) {
                //存在相同x不同y的点，无法使用多项式进行拟合
                return false;
            }
            inputMatrix[i][j] = inputMatrix[i][j] / base;
        }
        for(let j = 0; j < n; j++) {
            if(i != j) {
                var baseInner = inputMatrix[j][i];
                for (var k = 0; k < n + 1; k++) {
                    inputMatrix[j][k] = inputMatrix[j][k] - baseInner * inputMatrix[i][k];
                }
            }
        }
    }
    for (let i = 0; i < n; i++) {
        if (inputMatrix[i][n] > 0) {
            returnResult.push('+');
        }

        if (inputMatrix[i][n] != 0) {
            let tmp_x = '';
            for (let j = 0; j < n - 1 - i; j++) {
                tmp_x = tmp_x + "*x";
            }
            returnResult.push((inputMatrix[i][n] + tmp_x));
        }
    }
    return returnResult;
}