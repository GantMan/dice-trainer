import React from "react";
import ReactDOM from "react-dom";
import * as tf from "@tensorflow/tfjs";

import "./styles.css";

const inputShape = [12, 12, 1];
const epochs = 10
const diceData = require("./dice_data.json").dice;

// Wrap in a tidy for memory
const [stackedX, stackedY] = tf.tidy(() => {
  // Build a stacked tensor from JSON
  const xs = tf
    .concat([
      diceData.one,
      diceData.two,
      diceData.twor,
      diceData.three,
      diceData.threer,
      diceData.four,
      diceData.five,
      diceData.six,
      diceData.sixr,
    ])
    .expandDims(3);

  // Now the answers to their corresponding index
  const combo = [].concat(
    new Array(diceData.one.length).fill(0),
    new Array(diceData.two.length).fill(1),
    new Array(diceData.twor.length).fill(2),
    new Array(diceData.three.length).fill(3),
    new Array(diceData.threer.length).fill(4),
    new Array(diceData.four.length).fill(5),
    new Array(diceData.five.length).fill(6),
    new Array(diceData.six.length).fill(7),
    new Array(diceData.sixr.length).fill(8)
  );
  const ys = tf.oneHot(combo, 9);

  return [xs, ys];
});

const testModel = (model, img2Test) => {
  const img = new Image()
  img.crossOrigin = "anonymous";
  img.src = img2Test;
  img.onload = async () => {
    const imgTensor = tf.browser.fromPixels(img, 1).expandDims();

    const tensorResults = model.predict(imgTensor)
    const results = tensorResults.arraySync()

    console.log(`${img2Test} returned ${results}`)

  };
}

const doLinearPrediction = async () => {
  const model = tf.sequential();

  // model.add(tf.layers.conv2d({
  //   inputShape: inputShape,
  //   kernelSize: 5,
  //   filters: 8,
  //   strides: 1,
  //   activation: 'relu',
  //   kernelInitializer: 'varianceScaling'
  // }));

  // model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  model.add(tf.layers.flatten(
    {inputShape}
  ))

  model.add(
    tf.layers.dense({
      units: 64,
      inputShape: inputShape,
      activation: "relu",
    })
  );

  model.add(
    tf.layers.dense({
      units: 8,
      activation: "relu",
    })
  );

  model.add(
    tf.layers.dense({
      units: 9,
      kernelInitializer: "varianceScaling",
      activation: "softmax",
    })
  );

  const learningRate = 0.005;
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Make loss callback
  const printCallback = {
    onEpochEnd: (epoch, log) => {
      console.log(`${epoch+1} of ${epochs}`, log);
    },
  };

  await model.fit(stackedX, stackedY, {
    epochs,
    shuffle: true,
    batchSize: 32,
    callbacks: printCallback,
  });

  // Basic test
  testModel(model, "/1.png")
  testModel(model, "/2.png")
  testModel(model, "/3.png")
  testModel(model, "/4.png")
  testModel(model, "/5.png")
  testModel(model, "/6.png")
  testModel(model, "/7.png")
  testModel(model, "/8.png")
  testModel(model, "/9.png")

  // for save button
  window.model = model;
  return "DONE!"
};

class App extends React.Component {
  state = {
    simplePredict: "training model...",
  };

  componentDidMount() {
    doLinearPrediction().then((result) =>
      this.setState({ simplePredict: result })
    );
  }

  render() {
    return (
      <div className="App">
        <h1>Dice Trainer <span role="img" aria-label="die">🎲</span></h1>
        <img
          src="/ess70.png"
          width="150"
          alt="Gant cartoon"
        />
        <h3>
          <a href="http://gantlaborde.com/" >By Gant Laborde</a>
        </h3>
        <hr/>
        <h2>{this.state.simplePredict}</h2>
        <button
          onClick={async () => {
            if (!window.model) return;
            await window.model.save("downloads://dice-model");
          }}
        >
          Download Resulting Model
        </button>
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
