// app.js — Dual-Pooling Denoising Autoencoder (Max vs Avg)
import { 
    loadTrainFromFiles, loadTestFromFiles, splitTrainVal, 
    getRandomTestBatch, draw28x28ToCanvas, addNoise, calculatePSNR 
} from './data-loader.js'; 

console.log('========== DUAL POOLING AUTOENCODER ==========');
console.log('Backend:', tf.getBackend());

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
const maxRow = document.getElementById('maxRow');      
const avgRow = document.getElementById('avgRow');      
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
    maxRow.innerHTML = '';
    avgRow.innerHTML = '';    
    psnrDisplay.innerText = 'not evaluated';
    modelInfo.innerText = 'Model: not built';
    log('System reset.');
}

// ---------- Helper: Create canvas with image ----------
function createCanvasElement(tensor, label = null) {
    const canvas = document.createElement('canvas');
    canvas.width = 84; canvas.height = 84;
    draw28x28ToCanvas(tensor, canvas, 3);
    
    const cellDiv = document.createElement('div');
    cellDiv.className = 'canvas-cell';
    cellDiv.appendChild(canvas);
    
    if (label) {
        const labelDiv = document.createElement('div');
        labelDiv.className = 'psnr-label';
        labelDiv.innerText = label;
        cellDiv.appendChild(labelDiv);
    }
    
    return cellDiv;
}

// ---------- Build Dual-Pooling Autoencoder ----------
function buildDualPoolingAutoencoder() {
    tf.dispose(model);
    
    const input = tf.input({ shape: [28, 28, 1] });
    
    // Shared Encoder
    let x = tf.layers.conv2d({ 
        filters: 32, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal',
        name: 'shared_conv1'
    }).apply(input);
    x = tf.layers.batchNormalization({ name: 'shared_bn1' }).apply(x);
    
    x = tf.layers.conv2d({ 
        filters: 64, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal',
        name: 'shared_conv2'
    }).apply(x);
    x = tf.layers.batchNormalization({ name: 'shared_bn2' }).apply(x);
    
    const sharedFeatures = x;
    
    // ===== MAX POOLING PATH =====
    let maxPath = tf.layers.maxPooling2d({ poolSize: 2, padding: 'same', name: 'max_pool' }).apply(sharedFeatures);
    
    maxPath = tf.layers.conv2d({ 
        filters: 128, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal',
        name: 'max_conv1'
    }).apply(maxPath);
    maxPath = tf.layers.batchNormalization({ name: 'max_bn1' }).apply(maxPath);
    
    maxPath = tf.layers.conv2d({ 
        filters: 128, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal',
        name: 'max_conv2'
    }).apply(maxPath);
    maxPath = tf.layers.batchNormalization({ name: 'max_bn2' }).apply(maxPath);
    
    maxPath = tf.layers.upSampling2d({ size: [2, 2], name: 'max_upsample' }).apply(maxPath);
    
    maxPath = tf.layers.conv2d({ 
        filters: 64, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal',
        name: 'max_conv3'
    }).apply(maxPath);
    maxPath = tf.layers.batchNormalization({ name: 'max_bn3' }).apply(maxPath);
    
    const maxOutput = tf.layers.conv2d({ 
        filters: 1, 
        kernelSize: 3, 
        activation: 'sigmoid', 
        padding: 'same',
        kernelInitializer: 'glorotNormal',
        name: 'max_output'
    }).apply(maxPath);
    
    // ===== AVERAGE POOLING PATH =====
    let avgPath = tf.layers.averagePooling2d({ poolSize: 2, padding: 'same', name: 'avg_pool' }).apply(sharedFeatures);
    
    avgPath = tf.layers.conv2d({ 
        filters: 128, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal',
        name: 'avg_conv1'
    }).apply(avgPath);
    avgPath = tf.layers.batchNormalization({ name: 'avg_bn1' }).apply(avgPath);
    
    avgPath = tf.layers.conv2d({ 
        filters: 128, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal',
        name: 'avg_conv2'
    }).apply(avgPath);
    avgPath = tf.layers.batchNormalization({ name: 'avg_bn2' }).apply(avgPath);
    
    avgPath = tf.layers.upSampling2d({ size: [2, 2], name: 'avg_upsample' }).apply(avgPath);
    
    avgPath = tf.layers.conv2d({ 
        filters: 64, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal',
        name: 'avg_conv3'
    }).apply(avgPath);
    avgPath = tf.layers.batchNormalization({ name: 'avg_bn3' }).apply(avgPath);
    
    const avgOutput = tf.layers.conv2d({ 
        filters: 1, 
        kernelSize: 3, 
        activation: 'sigmoid', 
        padding: 'same',
        kernelInitializer: 'glorotNormal',
        name: 'avg_output'
    }).apply(avgPath);
    
    // Create model with two outputs
    model = tf.model({ 
        inputs: input, 
        outputs: [maxOutput, avgOutput] 
    });
    
    window.model = model;

    // Compile with losses for both outputs
    const optimizer = tf.train.adam(0.001);
    
    model.compile({
        optimizer: optimizer,
        loss: {
            max_output: 'meanSquaredError',
            avg_output: 'meanSquaredError'
        },
        metrics: {
            max_output: ['mae'],
            avg_output: ['mae']
        },
        loss_weights: {
            max_output: 0.5,
            avg_output: 0.5
        }
    });

    model.summary();
    console.log('Model outputs:', model.outputNames);

    log('🚀 DUAL-POOLING AUTOENCODER BUILT');
    log(`Total params: ${model.countParams()}`);
    modelInfo.innerText = `Dual-Pooling Model: ${model.countParams()} params`;
}

// ---------- Train Denoiser ----------
trainBtn.addEventListener('click', async () => {
    if (!model) {
        buildDualPoolingAutoencoder();
    }
    if (!valSplit.trainXs) {
        alert('Load data first.');
        return;
    }
    
    try {
        log('Preparing noisy training data...');
        
        const trainClean = valSplit.trainXs;
        const valClean = valSplit.valXs;
        
        const trainNoisy = addNoise(trainClean, NOISE_FACTOR);
        const valNoisy = addNoise(valClean, NOISE_FACTOR);
        
        log('Starting dual-pooling training...');
        const start = performance.now();
        
        const container = { name: 'Dual-Pooling Training', tab: 'Training' };
        const metrics = ['loss', 'val_loss'];
        const callbacks = tfvis.show.fitCallbacks(container, metrics, { zoomToFit: true });
        
        // Train with two outputs
        const history = await model.fit(trainNoisy, {
            max_output: trainClean,
            avg_output: trainClean
        }, {
            batchSize: 64,
            epochs: 15,
            validationData: [valNoisy, {
                max_output: valClean,
                avg_output: valClean
            }],
            shuffle: true,
            callbacks: {
                ...callbacks,
                onEpochEnd: async (epoch, logs) => {
                    tf.dispose();
                    await new Promise(resolve => setTimeout(resolve, 100));
                    log(`Epoch ${epoch+1}/15 - loss: ${logs.loss.toFixed(4)}`);
                }
            },
            verbose: 0
        });
        
        const duration = ((performance.now() - start)/1000).toFixed(2);
        log(`✅ Training complete in ${duration}s!`);
        timerSpan.innerText = `⏱️ ${duration}s`;
        
        tf.dispose([trainNoisy, valNoisy]);
        
    } catch (err) {
        log(`Training error: ${err.message}`);
    }
});

// ---------- Test Denoising with Both Pooling Types ----------
testFiveBtn.addEventListener('click', async () => {
    if (!model || !testTensors) {
        alert('Load model and test data first.');
        return;
    }
    
    try {
        const { xs: cleanBatch } = getRandomTestBatch(testTensors.xs, 5);
        const noisyBatch = addNoise(cleanBatch, NOISE_FACTOR);
        
        // Get both outputs
        const [maxDenoised, avgDenoised] = model.predict(noisyBatch);
        
        // Clear previous results - ИСПРАВЛЕНО!
        originalRow.innerHTML = '';
        noisyRow.innerHTML = '';
        maxRow.innerHTML = '';
        avgRow.innerHTML = '';
        
        // Create headers
        const originalHeader = document.createElement('div');
        originalHeader.className = 'row-label';
        originalHeader.innerText = '📌 Original images:';
        originalRow.appendChild(originalHeader);

        const noisyHeader = document.createElement('div');
        noisyHeader.className = 'row-label';
        noisyHeader.innerText = '🌫️ Noisy input:';
        noisyRow.appendChild(noisyHeader);

        const maxHeader = document.createElement('div');
        maxHeader.className = 'row-label';
        maxHeader.innerText = '🔷 Max Pooling Results:';
        maxRow.appendChild(maxHeader);                 

        const avgHeader = document.createElement('div');
        avgHeader.className = 'row-label';
        avgHeader.innerText = '🔶 Average Pooling Results:';
        avgRow.appendChild(avgHeader);                 
        
        // Create rows for images
        const originalRowDiv = document.createElement('div');
        originalRowDiv.className = 'canvas-row';
        originalRow.appendChild(originalRowDiv);
        
        const noisyRowDiv = document.createElement('div');
        noisyRowDiv.className = 'canvas-row';
        noisyRow.appendChild(noisyRowDiv);
        
        const maxRowDiv = document.createElement('div');
        maxRowDiv.className = 'canvas-row';
        maxRow.appendChild(maxRowDiv);                  

        const avgRowDiv = document.createElement('div');
        avgRowDiv.className = 'canvas-row';
        avgRow.appendChild(avgRowDiv);                  
        
        let totalPSNRMax = 0, totalPSNRAvg = 0;
        
        for (let i = 0; i < 5; i++) {
            const cleanImg = cleanBatch.slice([i,0,0,0], [1,28,28,1]).squeeze();
            const noisyImg = noisyBatch.slice([i,0,0,0], [1,28,28,1]).squeeze();
            const maxImg = maxDenoised.slice([i,0,0,0], [1,28,28,1]).squeeze();
            const avgImg = avgDenoised.slice([i,0,0,0], [1,28,28,1]).squeeze();
            
            const psnrMax = calculatePSNR(cleanImg, maxImg);
            const psnrAvg = calculatePSNR(cleanImg, avgImg);
            totalPSNRMax += psnrMax;
            totalPSNRAvg += psnrAvg;
            
            originalRowDiv.appendChild(createCanvasElement(cleanImg));
            noisyRowDiv.appendChild(createCanvasElement(noisyImg));
            
            const maxCell = createCanvasElement(maxImg, `Max: ${psnrMax.toFixed(1)}dB`);
            maxRowDiv.appendChild(maxCell);
            
            const avgCell = createCanvasElement(avgImg, `Avg: ${psnrAvg.toFixed(1)}dB`);
            avgRowDiv.appendChild(avgCell);
            
            tf.dispose([cleanImg, noisyImg, maxImg, avgImg]);
        }
        
        const avgPSNRMax = (totalPSNRMax/5).toFixed(2);
        const avgPSNRAvg = (totalPSNRAvg/5).toFixed(2);
        
        psnrDisplay.innerHTML = `
            Max Pooling: ${avgPSNRMax} dB | 
            Average Pooling: ${avgPSNRAvg} dB
        `;
        
        log(`📊 Max PSNR: ${avgPSNRMax} dB | Avg PSNR: ${avgPSNRAvg} dB`);
        
        tf.dispose([cleanBatch, noisyBatch, maxDenoised, avgDenoised]);
        
    } catch (err) {
        log(`Test error: ${err.message}`);
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

        if (!model) buildDualPoolingAutoencoder();
    } catch (err) {
        log(`Error loading data: ${err.message}`);
    }
});

// ---------- Build Model Button ----------
buildModelBtn.addEventListener('click', buildDualPoolingAutoencoder);

// ---------- Save Model ----------
saveModelBtn.addEventListener('click', async () => {
    if (!model) { alert('No model to save'); return; }
    try {
        await model.save('downloads://dual-pooling-autoencoder');
        log('Dual-pooling model download initiated.');
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
        
        const optimizer = tf.train.adam(0.001);
        model.compile({
            optimizer: optimizer,
            loss: {
                max_output: 'meanSquaredError',
                avg_output: 'meanSquaredError'
            },
            metrics: {
                max_output: ['mae'],
                avg_output: ['mae']
            },
            loss_weights: {
                max_output: 0.5,
                avg_output: 0.5
            }
        });
        
        log('✅ Dual-pooling model loaded!');
        modelInfo.innerText = `Dual model loaded: ${model.countParams()} params`;
    } catch (err) {
        log(`Load error: ${err.message}`);
    }
});

// ---------- Toggle Visor ----------
toggleVisorBtn.addEventListener('click', () => tfvis.visor().toggle());

// ---------- Reset ----------
resetBtn.addEventListener('click', resetAll);

// Initial setup
buildDualPoolingAutoencoder();
log('✨ Dual-Pooling Autoencoder ready!');