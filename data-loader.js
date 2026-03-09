// data-loader.js — CSV parsing with noise addition for denoising autoencoder

export async function loadTrainFromFiles(file) {
    if (!file) throw new Error('No train file provided');
    return await parseCSVFile(file);
}

export async function loadTestFromFiles(file) {
    if (!file) throw new Error('No test file provided');
    return await parseCSVFile(file);
}

async function parseCSVFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const text = e.target.result;
                const lines = text.split(/\r?\n/).filter(line => line.trim() !== '');
                if (lines.length === 0) throw new Error('CSV file is empty');

                const labels = [];
                const pixels = [];

                for (let line of lines) {
                    const values = line.split(',').map(Number);
                    if (values.length !== 785) continue;
                    const label = values[0];
                    if (isNaN(label) || label < 0 || label > 9) continue;
                    labels.push(label);
                    pixels.push(values.slice(1));
                }

                if (labels.length === 0) throw new Error('No valid rows found');

                const xs = tf.tidy(() => {
                    const pixelTensor = tf.tensor2d(pixels, [labels.length, 784]);
                    const reshaped = pixelTensor.reshape([labels.length, 28, 28, 1]);
                    return reshaped.div(255.0);
                });

                const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 10).toFloat();

                resolve({ xs, ys, labels: tf.tensor1d(labels, 'int32') });
            } catch (err) {
                reject(err);
            }
        };
        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

export function splitTrainVal(xs, ys, valRatio = 0.1) {
    const total = xs.shape[0];
    const valCount = Math.floor(total * valRatio);
    const trainCount = total - valCount;

    return tf.tidy(() => {
        const trainXs = xs.slice([0, 0, 0, 0], [trainCount, 28, 28, 1]);
        const trainYs = ys.slice([0, 0], [trainCount, 10]);
        const valXs = xs.slice([trainCount, 0, 0, 0], [valCount, 28, 28, 1]);
        const valYs = ys.slice([trainCount, 0], [valCount, 10]);

        return { trainXs, trainYs, valXs, valYs };
    });
}

/**
 * Add Gaussian noise to images
 * @param {tf.Tensor} images - tensor of shape [N,28,28,1] with values [0,1]
 * @param {number} noiseFactor - standard deviation of noise (e.g., 0.2)
 * @returns {tf.Tensor} noisy images clipped to [0,1]
 */
export function addNoise(images, noiseFactor = 0.2) {
    return tf.tidy(() => {
        const noise = tf.randomNormal(images.shape, 0, noiseFactor);
        const noisy = images.add(noise);
        return noisy.clipByValue(0, 1);
    });
}

/**
 * Calculate PSNR between two images
 * @param {tf.Tensor} img1 - original clean image [28,28,1]
 * @param {tf.Tensor} img2 - denoised image [28,28,1]
 * @returns {number} PSNR in dB
 */
export function calculatePSNR(img1, img2) {
    return tf.tidy(() => {
        const mse = img1.sub(img2).square().mean().dataSync()[0];
        if (mse === 0) return Infinity;
        const maxPixel = 1.0;
        return 10 * Math.log10((maxPixel * maxPixel) / mse);
    });
}

export function getRandomTestBatch(xs, k = 5) {
    const total = xs.shape[0];
    if (total === 0) throw new Error('Empty test set');
    const indices = [];
    for (let i = 0; i < k; i++) {
        indices.push(Math.floor(Math.random() * total));
    }
    return tf.tidy(() => {
        const batchXs = tf.gather(xs, indices);
        return { xs: batchXs, indices };
    });
}

export function draw28x28ToCanvas(tensor, canvas, scale = 3) {
    return tf.tidy(() => {
        const data = tensor.dataSync();
        const ctx = canvas.getContext('2d');
        const width = 28 * scale, height = 28 * scale;
        canvas.width = width; canvas.height = height;
        ctx.clearRect(0, 0, width, height);

        const imageData = ctx.createImageData(width, height);
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const val = data[y * 28 + x] * 255;
                const gray = Math.min(255, Math.max(0, Math.round(val)));
                for (let dy = 0; dy < scale; dy++) {
                    for (let dx = 0; dx < scale; dx++) {
                        const px = x * scale + dx;
                        const py = y * scale + dy;
                        const idx = (py * width + px) * 4;
                        imageData.data[idx] = gray;
                        imageData.data[idx + 1] = gray;
                        imageData.data[idx + 2] = gray;
                        imageData.data[idx + 3] = 255;
                    }
                }
            }
        }
        ctx.putImageData(imageData, 0, 0);
    });
}