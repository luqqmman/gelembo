const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');
    
async function predictClassification(model, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat()
    
        const classes = ['Melanocytic nevus', 'Squamous cell carcinoma', 'Vascular lesion'];
    
        const prediction = model.predict(tensor);
        const score = await prediction.data();

        const label = score[0] > 0.5 ? 'Cancer' : 'Non-cancer';
        const suggestion = score[0] > 0.5 ? 'Segera periksa ke dokter!' : 'Anda sehat';

    
        return { label, suggestion };
    } catch (error) {
        throw new InputError(`Terjadi kesalahan input: ${error.message}`)
    }
}
    
module.exports = predictClassification;