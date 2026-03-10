// app.js — Denoising Autoencoder with max/avg pooling comparison
import { 
    loadTrainFromFiles, loadTestFromFiles, splitTrainVal, 
    getRandomTestBatch, draw28x28ToCanvas, addNoise, calculatePSNR 
} from './data-loader.js'; 

console.log('========== DEBUG INFO ==========');
console.log('Backend:', tf.getBackend());
console.log('Memory:', tf.memory());

// State
let trainTensors = null;
let testTensors = null;
let model = null;
let valSplit = { trainXs: null, trainYs: null, valXs: null, valYs: null };

// UI Elements
const trainFile = document.getElementById('trainFile');
const testFile = document.getElementById('testFile');
const dataStatus = document.getElementById('dataStatus');
const logDiv = document.getElementById('log');
const timerSpan = document.getElementById('timerSpan');
const originalRow = document.getElementById('originalRow');
const noisyRow = document.getElementById('noisyRow');
const denoisedRow = document.getElementById('denoisedRow');
const psnrDisplay = document.getElementById('psnrDisplay');
const modelInfo = document.getElementById('modelInfo');

// Buttons
const loadDataBtn = document.getElementById('loadDataBtn');
const resetBtn = document.getElementById('resetBtn');
const buildModelBtn = document.getElementById('buildModelBtn');
const trainBtn = document.getElementById('trainBtn');
const testFiveBtn = document.getElementById('testFiveBtn');
const toggleVisorBtn = document.getElementById('toggleVisorBtn');
const saveModelBtn = document.getElementById('saveModelBtn');
const loadModelBtn = document.getElementById('loadModelBtn');
const loadJsonFile = document.getElementById('loadJsonFile');
const loadBinFile = document.getElementById('loadBinFile');

// Pooling selection
const poolingRadios = document.getElementsByName('pooling');
let currentPooling = 'max'; // default

// Listen for pooling changes
poolingRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
        if (e.target.checked) {
            currentPooling = e.target.value;
            log(`Pooling type set to: ${currentPooling}`);
        }
    });
});

// Constants
const NOISE_FACTOR = 0.25; // 25% noise

function log(message) {
    logDiv.innerText += '\n' + message;
    logDiv.scrollTop = logDiv.scrollHeight;
    console.log(message);
}

function clearLog() { logDiv.innerText = ''; }

function resetAll() {
    tf.dispose([trainTensors?.xs, trainTensors?.ys, testTensors?.xs, testTensors?.ys,
                valSplit.trainXs, valSplit.trainYs, valSplit.valXs, valSplit.valYs]);
    if (model) { model.dispose(); model = null; }
    trainTensors = null; testTensors = null;
    valSplit = { trainXs: null, trainYs: null, valXs: null, valYs: null };
    dataStatus.innerText = 'Data cleared.';
    originalRow.innerHTML = '<div style="color:#64748b;">— reset —</div>';
    noisyRow.innerHTML = '';
    denoisedRow.innerHTML = '';
    psnrDisplay.innerText = 'not evaluated';
    modelInfo.innerText = 'Model: not built';
    log('System reset.');
}

// ---------- Build Denoising Autoencoder with BatchNormalization ----------
function buildDenoisingAutoencoder(poolingType = 'max') {
    tf.dispose(model);
    
    const input = tf.input({ shape: [28, 28, 1] });
    
    // Encoder with BatchNorm
    let x = tf.layers.conv2d({ 
        filters: 32, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal'
    }).apply(input);
    x = tf.layers.batchNormalization().apply(x);
    
    x = tf.layers.conv2d({ 
        filters: 64, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal'
    }).apply(x);
    x = tf.layers.batchNormalization().apply(x);
    
    // Pooling layer (max or average)
    if (poolingType === 'max') {
        x = tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' }).apply(x);
    } else {
        x = tf.layers.averagePooling2d({ poolSize: 2, padding: 'same' }).apply(x);
    }
    
    // Bottleneck with BatchNorm
    x = tf.layers.conv2d({ 
        filters: 128, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal'
    }).apply(x);
    x = tf.layers.batchNormalization().apply(x);
    
    x = tf.layers.conv2d({ 
        filters: 128, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal'
    }).apply(x);
    x = tf.layers.batchNormalization().apply(x);
    
    // Decoder - Upsampling
    x = tf.layers.upSampling2d({ size: [2, 2] }).apply(x);
    
    x = tf.layers.conv2d({ 
        filters: 64, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal'
    }).apply(x);
    x = tf.layers.batchNormalization().apply(x);
    
    x = tf.layers.conv2d({ 
        filters: 32, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal'
    }).apply(x);
    x = tf.layers.batchNormalization().apply(x);
    
    // Output layer with sigmoid for [0,1] range
    const output = tf.layers.conv2d({ 
        filters: 1, 
        kernelSize: 3, 
        activation: 'sigmoid', 
        padding: 'same',
        kernelInitializer: 'glorotNormal'
    }).apply(x);
    
    model = tf.model({ inputs: input, outputs: output });
    
    window.model = model;

    // Explicit learning rate for Adam
    const optimizer = tf.train.adam(0.01);
    
    model.compile({
        optimizer: optimizer,
        loss: 'meanSquaredError',
        metrics: ['mae']
    });

    model.summary();
    console.log('Model output shape:', model.outputShape);

    log(`Denoising autoencoder built with ${poolingType} pooling and BatchNormalization.`);
    model.summary(null, null, (msg) => log(msg));
    modelInfo.innerText = `Denoiser (${poolingType} pool + BN): ${model.countParams()} params`;
}

// ---------- Train Denoiser ----------
trainBtn.addEventListener('click', async () => {
    if (!model) {
        buildDenoisingAutoencoder(currentPooling);
    }
    if (!valSplit.trainXs) {
        alert('Load data first.');
        return;
    }
    
    try {
        log('Preparing noisy training data...');
        
        // Create noisy versions of training and validation images
        const trainClean = valSplit.trainXs;
        const valClean = valSplit.valXs;
        
        // Add noise
        const trainNoisy = addNoise(trainClean, NOISE_FACTOR);
        const valNoisy = addNoise(valClean, NOISE_FACTOR);
        
        log('Starting denoiser training...');
        const start = performance.now();
        
        const container = { name: 'Denoising Training', tab: 'Training' };
        const metrics = ['loss', 'val_loss', 'mae', 'val_mae'];
        const callbacks = tfvis.show.fitCallbacks(container, metrics, { zoomToFit: true });
        
        const history = await model.fit(trainNoisy, trainClean, {
            batchSize: 128,
            epochs: 15,
            validationData: [valNoisy, valClean],
            shuffle: true,
            callbacks: callbacks,
            verbose: 0
        });
        
        const duration = ((performance.now() - start)/1000).toFixed(2);
        const lastLoss = history.history.val_loss[history.history.val_loss.length-1];
        log(`✅ Denoiser trained in ${duration}s. Final val loss: ${lastLoss.toFixed(4)}`);
        timerSpan.innerText = `⏱️ ${duration}s`;
        
        // Clean up
        tf.dispose([trainNoisy, valNoisy]);
        
    } catch (err) {
        log(`Training error: ${err.message}`);
    }
});

// ---------- Test Denoising on 5 Random Images ----------
testFiveBtn.addEventListener('click', async () => {
    if (!model || !testTensors) {
        alert('Load model and test data first.');
        return;
    }
    
    try {
        // Get 5 random clean images
        const { xs: cleanBatch, indices } = getRandomTestBatch(testTensors.xs, 5);
        
        // Create noisy versions
        const noisyBatch = addNoise(cleanBatch, NOISE_FACTOR);
        
        // Run denoising
        const denoisedBatch = model.predict(noisyBatch);
        
        // Clear previous results
        originalRow.innerHTML = '';
        noisyRow.innerHTML = '';
        denoisedRow.innerHTML = '';
        
        let totalPSNR = 0;
        
        // Display each image
        for (let i = 0; i < 5; i++) {
            // Extract individual images
            const cleanImg = cleanBatch.slice([i,0,0,0], [1,28,28,1]).squeeze();
            const noisyImg = noisyBatch.slice([i,0,0,0], [1,28,28,1]).squeeze();
            const denoisedImg = denoisedBatch.slice([i,0,0,0], [1,28,28,1]).squeeze();
            
            // Calculate PSNR
            const psnr = calculatePSNR(cleanImg, denoisedImg);
            totalPSNR += psnr;
            
            // Create canvas elements
            const createCanvas = (tensor, container) => {
                const canvas = document.createElement('canvas');
                canvas.width = 84; canvas.height = 84;
                draw28x28ToCanvas(tensor, canvas, 3);
                
                const cellDiv = document.createElement('div');
                cellDiv.className = 'canvas-cell';
                cellDiv.appendChild(canvas);
                
                // Add PSNR label for denoised images
                if (container === denoisedRow) {
                    const psnrDiv = document.createElement('div');
                    psnrDiv.className = 'psnr-label';
                    psnrDiv.innerText = `PSNR: ${psnr.toFixed(1)}dB`;
                    cellDiv.appendChild(psnrDiv);
                }
                
                container.appendChild(cellDiv);
            };
            
            createCanvas(cleanImg, originalRow);
            createCanvas(noisyImg, noisyRow);
            createCanvas(denoisedImg, denoisedRow);
            
            tf.dispose([cleanImg, noisyImg, denoisedImg]);
        }
        
        const avgPSNR = totalPSNR / 5;
        psnrDisplay.innerText = `Average PSNR: ${avgPSNR.toFixed(2)} dB (${currentPooling} pooling)`;
        log(`Average PSNR on 5 test images: ${avgPSNR.toFixed(2)} dB`);
        
        tf.dispose([cleanBatch, noisyBatch, denoisedBatch]);
        
    } catch (err) {
        log(`Denoising test error: ${err.message}`);
    }
});

// ---------- Load Data ----------
loadDataBtn.addEventListener('click', async () => {
    if (!trainFile.files[0] || !testFile.files[0]) {
        alert('Please select both train and test CSV files.');
        return;
    }
    try {
        resetAll();
        clearLog();
        log('Loading train file...');
        trainTensors = await loadTrainFromFiles(trainFile.files[0]);
        log(`Train samples: ${trainTensors.xs.shape[0]}`);
        log('Loading test file...');
        testTensors = await loadTestFromFiles(testFile.files[0]);
        log(`Test samples: ${testTensors.xs.shape[0]}`);

        const split = splitTrainVal(trainTensors.xs, trainTensors.ys, 0.1);
        valSplit = split;
        log(`Validation split: ${split.valXs.shape[0]} samples`);

        dataStatus.innerHTML = `✅ Train: ${trainTensors.xs.shape[0]} | Test: ${testTensors.xs.shape[0]} | Val: ${split.valXs.shape[0]}`;

        // Auto-build model if none exists
        if (!model) buildDenoisingAutoencoder(currentPooling);
    } catch (err) {
        log(`Error loading data: ${err.message}`);
        console.error(err);
    }
});

// ---------- Build Model Button ----------
buildModelBtn.addEventListener('click', () => {
    buildDenoisingAutoencoder(currentPooling);
});

// ---------- Save Model ----------
saveModelBtn.addEventListener('click', async () => {
    if (!model) { alert('No model to save'); return; }
    try {
        await model.save('downloads://denoising-autoencoder');
        log('Denoiser model download initiated.');
    } catch (err) {
        log(`Save error: ${err.message}`);
    }
});

// ---------- Load Model ----------
loadModelBtn.addEventListener('click', async () => {
    if (!loadJsonFile.files[0] || !loadBinFile.files[0]) {
        alert('Please select both model.json and weights.bin');
        return;
    }
    try {
        const jsonFile = loadJsonFile.files[0];
        const binFile = loadBinFile.files[0];
        const loadedModel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
        if (model) model.dispose();
        model = loadedModel;
        
        // Recompile with explicit optimizer
        const optimizer = tf.train.adam(0.01);
        model.compile({ 
            optimizer: optimizer, 
            loss: 'meanSquaredError', 
            metrics: ['mae'] 
        });
        
        log('Denoiser model loaded from files. Re-compiled with Adam(0.01).');
        
        // Try to infer pooling type from architecture (simple heuristic)
        const config = model.toJSON(null, false);
        let poolingType = 'unknown';
        if (JSON.stringify(config).includes('MaxPooling2D')) poolingType = 'max';
        else if (JSON.stringify(config).includes('AveragePooling2D')) poolingType = 'avg';
        
        modelInfo.innerText = `Denoiser loaded (${poolingType} pool): ${model.countParams()} params`;
    } catch (err) {
        log(`Load model error: ${err.message}`);
    }
});

// ========== НОВЫЙ КОД: ЗАГРУЗКА МОДЕЛИ С GITHUB ==========
// ---------- Load Pre-trained Model from GitHub ----------
async function loadGitHubModel() {
    try {
        log('Loading pre-trained model from GitHub...');
        
        // Raw ссылки на файлы модели
        const modelUrl = 'https://raw.githubusercontent.com/GeorgeRudenko/NNDLHW4/main/trained-model/model.json';
        // Загружаем модель
        const loadedModel = await tf.loadLayersModel(modelUrl);
        
        if (window.model) {
            window.model.dispose();
        }
        
        window.model = loadedModel;
        
        // Компилируем с теми же параметрами
        const optimizer = tf.train.adam(0.01);
        loadedModel.compile({
            optimizer: optimizer,
            loss: 'meanSquaredError',
            metrics: ['mae']
        });
        
        log('✅ Classroom model loaded successfully!');
        log(`Model params: ${loadedModel.countParams()}`);
        modelInfo.innerText = `Classroom model loaded: ${loadedModel.countParams()} params`;
        
        // Определяем тип pooling из архитектуры
        const config = loadedModel.toJSON(null, false);
        let poolingType = 'unknown';
        if (JSON.stringify(config).includes('MaxPooling2D')) poolingType = 'max';
        else if (JSON.stringify(config).includes('AveragePooling2D')) poolingType = 'avg';
        
        log(`Pooling type: ${poolingType}`);
        
    } catch (err) {
        log(`❌ Error loading model: ${err.message}`);
        console.error(err);
    }
}

// Добавляем обработчик для новой кнопки
document.getElementById('loadGitHubModelBtn').addEventListener('click', loadGitHubModel);
// ========== КОНЕЦ НОВОГО КОДА ==========

// ---------- Toggle Visor ----------
toggleVisorBtn.addEventListener('click', () => tfvis.visor().toggle());

// ---------- Reset ----------
resetBtn.addEventListener('click', resetAll);

// Initial setup
buildDenoisingAutoencoder('max');
log('Denoising Autoencoder ready. Load MNIST data and start training!');