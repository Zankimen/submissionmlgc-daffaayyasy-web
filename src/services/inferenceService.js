const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');


async function predictClassification(model, image) {
    try {
        const tensor = tf.node 
            .decodeJpeg(image) 
            .resizeNearestNeighbor([224, 224]) 
            .expandDims() 
            .toFloat() 
 
        const classes = ['Cancer', 'Non-cancer'];
        const suggestionArr = ['Segera periksa ke dokter!', 'Penyakit kanker tidak terdeteksi.'];
 
        const prediction = model.predict(tensor);
        const score = await prediction.data(); 
        
        const confidenceScore = Math.max(...score) * 100; 
 
        
        const label = confidenceScore > 99 ? classes[0] : classes[1];

        let suggestion;
 
        if(label === classes[0]) {
            suggestion = suggestionArr[0];
        }
 
        if(label === classes[1]) {
            suggestion = suggestionArr[1];
        }
 
        return { confidenceScore, label, suggestion };
    } catch (error) {
        throw new InputError('Terjadi kesalahan dalam melakukan prediksi');
    }
}
 
module.exports = predictClassification;