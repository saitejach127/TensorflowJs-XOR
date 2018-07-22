const model = tf.sequential();
model.add(tf.layers.dense({
	inputShape: [2],
	kernelSize: 5,
	activation: 'relu',
	units: 10
}));
model.add(tf.layers.dense({
	units: 1,
	activation: 'relu'
}));

const LEARNING_RATE = 0.4;
const optimizer = tf.train.sgd(LEARNING_RATE);
model.compile({
	optimizer: optimizer,
	loss: 'meanSquaredError'
});
const xs = tf.tensor([[1,1],[1,0],[0,1],[0,0]]);
console.log("input");
xs.print();
const ys = tf.tensor([[0],[1],[1],[0]]);
console.log("output");
ys.print();

console.log("training");

model.fit(xs,ys,{
	epochs: 50,
	shuffle: true
}).then((history) => {
	prediction = model.predict(tf.tensor([[1,1],[1,0],[0,1],[0,0]]));
	prediction.print();
	console.log("loss1 " + history.history.loss[0]);
	model.fit(xs,ys,{
	epochs:20,
	shuffle:true
}).then((history) => {
	const predic = model.predict(tf.tensor([[1,1],[1,0],[0,1],[0,0]]));
	predic.print();
	console.log("loss2 " + history.history.loss[0]);
});
});
