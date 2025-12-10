// API Configuration
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5000/api'
    : 'https://your-backend-url.onrender.com/api'; // Update with your Render URL

// Global state
let state = {
    datasetId: null,
    analysis: null,
    preprocessingSteps: [],
    processedFilename: null,
    results: null
};

// DOM Elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const analysisSection = document.getElementById('analysis-section');
const analysisContent = document.getElementById('analysis-content');
const preprocessSection = document.getElementById('preprocess-section');
const reportSection = document.getElementById('report-section');
const reportContent = document.getElementById('report-content');
const trainingSection = document.getElementById('training-section');
const resultsSection = document.getElementById('results-section');
const downloadSection = document.getElementById('download-section');
const targetColumnSelect = document.getElementById('target-column');
const progressContainer = document.getElementById('progress-container');
const progressFill = document.getElementById('progress-fill');
const progressPercentage = document.getElementById('progress-percentage');
const progressMessage = document.getElementById('progress-message');
const downloadBtn = document.getElementById('download-btn');

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // File upload handlers
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    
    // Button handlers
    document.getElementById('apply-defaults').addEventListener('click', applySmartDefaults);
    document.getElementById('apply-preprocessing').addEventListener('click', applyPreprocessing);
    document.getElementById('train-models').addEventListener('click', trainModels);
    downloadBtn.addEventListener('click', downloadProcessedDataset);
});

// File Upload Functions
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.style.borderColor = '#7c3aed';
    uploadArea.style.background = '#f1f5f9';
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.style.borderColor = '#4f46e5';
    uploadArea.style.background = '#f8fafc';
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.style.borderColor = '#4f46e5';
    uploadArea.style.background = '#f8fafc';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

async function processFile(file) {
    showStatus('Uploading dataset...', 'info');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            state.datasetId = data.dataset_id;
            state.analysis = data.analysis;
            
            showStatus('Dataset uploaded successfully!', 'success');
            displayAnalysis(data.analysis);
            setupPreprocessingControls(data.analysis);
            setupTrainingControls(data.analysis);
            
            // Show sections
            analysisSection.style.display = 'block';
            preprocessSection.style.display = 'block';
            reportSection.style.display = 'none';
            trainingSection.style.display = 'block';
            resultsSection.style.display = 'none';
            downloadSection.style.display = 'none';
            
            // Scroll to analysis
            analysisSection.scrollIntoView({ behavior: 'smooth' });
        } else {
            showStatus(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatus(`Upload failed: ${error.message}`, 'error');
    }
}

function showStatus(message, type = 'info') {
    uploadStatus.textContent = message;
    uploadStatus.className = `status-message ${type}`;
    uploadStatus.style.display = 'block';
}

// Analysis Display
function displayAnalysis(analysis) {
    const { shape, columns, missing_percentage, data_types } = analysis;
    
    let html = `
        <div class="analysis-card">
            <h3><i class="fas fa-table"></i> Dataset Overview</h3>
            <div class="stat-number">${shape[0]} × ${shape[1]}</div>
            <p>${shape[0]} rows, ${shape[1]} columns</p>
        </div>
        
        <div class="analysis-card">
            <h3><i class="fas fa-columns"></i> Column Types</h3>
            <div class="stat-number">${analysis.numerical_cols.length}</div>
            <p>Numerical columns</p>
            <div class="stat-number">${analysis.categorical_cols.length}</div>
            <p>Categorical columns</p>
        </div>
        
        <div class="analysis-card">
            <h3><i class="fas fa-exclamation-triangle"></i> Missing Values</h3>
            <div class="stat-number">
                ${Object.values(missing_percentage).filter(p => p > 0).length}
            </div>
            <p>Columns with missing values</p>
        </div>
    `;
    
    // Add column details
    html += '<div class="analysis-card" style="grid-column: 1 / -1;">';
    html += '<h3><i class="fas fa-list"></i> Column Details</h3>';
    html += '<div style="max-height: 300px; overflow-y: auto;">';
    html += '<table style="width: 100%; border-collapse: collapse;">';
    html += '<tr><th>Column</th><th>Type</th><th>Missing %</th><th>Unique Values</th></tr>';
    
    columns.forEach(col => {
        const missingPercent = missing_percentage[col.name];
        const uniqueCount = col.unique_values !== null ? col.unique_values : 'N/A';
        
        html += `
            <tr>
                <td><strong>${col.name}</strong></td>
                <td><span class="column-type">${col.type}</span></td>
                <td>
                    <span style="color: ${missingPercent > 50 ? '#ef4444' : missingPercent > 0 ? '#f59e0b' : '#10b981'}">
                        ${missingPercent}%
                    </span>
                </td>
                <td>${uniqueCount}</td>
            </tr>
        `;
    });
    
    html += '</table></div></div>';
    
    analysisContent.innerHTML = html;
}

// Preprocessing Controls
function setupPreprocessingControls(analysis) {
    const typeContainer = document.getElementById('type-conversion-container');
    const missingContainer = document.getElementById('missing-values-container');
    const encodingContainer = document.getElementById('encoding-container');
    
    typeContainer.innerHTML = '';
    missingContainer.innerHTML = '';
    encodingContainer.innerHTML = '';
    
    analysis.columns.forEach(col => {
        const { name, type, missing_percent } = col;
        
        // Data type conversion
        if (type !== 'float64' && type !== 'int64') {
            const typeHtml = `
                <div class="column-control">
                    <div class="column-header">
                        <span class="column-name">${name}</span>
                        <span class="column-type">${type}</span>
                    </div>
                    <div class="column-controls">
                        <select class="form-control type-select" data-column="${name}">
                            <option value="">Keep as is</option>
                            <option value="numeric">Convert to Numeric</option>
                            <option value="datetime">Convert to Datetime</option>
                            <option value="category">Convert to Category</option>
                        </select>
                    </div>
                </div>
            `;
            typeContainer.innerHTML += typeHtml;
        }
        
        // Missing values handling (for columns with < 50% missing)
        if (missing_percent > 0 && missing_percent <= 50) {
            const missingHtml = `
                <div class="column-control">
                    <div class="column-header">
                        <span class="column-name">${name}</span>
                        <span class="column-type">Missing: ${missing_percent}%</span>
                    </div>
                    <div class="column-controls">
                        <select class="form-control missing-select" data-column="${name}">
                            <option value="">Select strategy</option>
                            ${type === 'object' || type === 'category' ? `
                                <option value="mode">Fill with Mode</option>
                            ` : `
                                <option value="mean">Fill with Mean</option>
                                <option value="median">Fill with Median</option>
                                <option value="mode">Fill with Mode</option>
                            `}
                            <option value="drop">Remove rows</option>
                        </select>
                    </div>
                </div>
            `;
            missingContainer.innerHTML += missingHtml;
        }
        
        // Encoding for categorical columns
        if (analysis.categorical_cols.includes(name)) {
            const encodingHtml = `
                <div class="column-control">
                    <div class="column-header">
                        <span class="column-name">${name}</span>
                        <span class="column-type">Categorical</span>
                    </div>
                    <div class="column-controls">
                        <select class="form-control encoding-select" data-column="${name}">
                            <option value="">No encoding</option>
                            <option value="label">Label Encoding</option>
                            <option value="onehot">One-Hot Encoding</option>
                        </select>
                    </div>
                </div>
            `;
            encodingContainer.innerHTML += encodingHtml;
        }
    });
}

function applySmartDefaults() {
    // Apply sensible defaults for preprocessing
    document.querySelectorAll('.type-select').forEach(select => {
        const colName = select.dataset.column;
        const colInfo = state.analysis.columns.find(c => c.name === colName);
        
        if (colInfo) {
            if (colInfo.type === 'object' && colInfo.unique_values < 10) {
                select.value = 'category';
            } else if (colInfo.type.includes('date') || colInfo.type.includes('time')) {
                select.value = 'datetime';
            }
        }
    });
    
    document.querySelectorAll('.missing-select').forEach(select => {
        if (select.value === '') {
            select.value = 'mode';
        }
    });
    
    document.querySelectorAll('.encoding-select').forEach(select => {
        if (select.value === '') {
            select.value = 'label';
        }
    });
    
    showStatus('Smart defaults applied! Review and adjust if needed.', 'success');
}

async function applyPreprocessing() {
    // Collect preprocessing options
    const options = {
        type_conversions: {},
        missing_value_strategy: {},
        encoding: {},
        remove_outliers: document.getElementById('remove-outliers').checked
    };
    
    // Collect type conversions
    document.querySelectorAll('.type-select').forEach(select => {
        if (select.value) {
            options.type_conversions[select.dataset.column] = select.value;
        }
    });
    
    // Collect missing value strategies
    document.querySelectorAll('.missing-select').forEach(select => {
        if (select.value) {
            options.missing_value_strategy[select.dataset.column] = select.value;
        }
    });
    
    // Collect encoding options
    document.querySelectorAll('.encoding-select').forEach(select => {
        if (select.value) {
            options.encoding[select.dataset.column] = select.value;
        }
    });
    
    try {
        showStatus('Applying preprocessing...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/preprocess`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(options)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            state.preprocessingSteps = data.steps;
            state.processedFilename = data.processed_filename;
            
            showStatus('Preprocessing completed successfully!', 'success');
            displayReport(data.steps, data.processed_analysis);
            
            // Show report and enable training
            reportSection.style.display = 'block';
            reportSection.scrollIntoView({ behavior: 'smooth' });
        } else {
            showStatus(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatus(`Preprocessing failed: ${error.message}`, 'error');
    }
}

function displayReport(steps, processedAnalysis) {
    let html = '<h3>Preprocessing Steps Applied:</h3>';
    
    steps.forEach((step, index) => {
        html += `
            <div class="report-step">
                <i class="fas fa-check-circle"></i>
                <span>${step}</span>
            </div>
        `;
    });
    
    if (processedAnalysis) {
        html += '<h3 style="margin-top: 20px;">Processed Dataset Analysis:</h3>';
        html += `
            <div class="report-step">
                <i class="fas fa-chart-bar"></i>
                <span>New shape: ${processedAnalysis.shape[0]} rows × ${processedAnalysis.shape[1]} columns</span>
            </div>
            <div class="report-step">
                <i class="fas fa-check"></i>
                <span>Columns with missing values: ${Object.values(processedAnalysis.missing_percentage).filter(p => p > 0).length}</span>
            </div>
        `;
    }
    
    reportContent.innerHTML = html;
}

// Training Controls
function setupTrainingControls(analysis) {
    // Populate target column dropdown
    targetColumnSelect.innerHTML = '<option value="">Select target column (optional)</option>';
    
    analysis.columns.forEach(col => {
        const option = document.createElement('option');
        option.value = col.name;
        option.textContent = `${col.name} (${col.type})`;
        targetColumnSelect.appendChild(option);
    });
}

async function trainModels() {
    // Collect training options
    const selectedModels = Array.from(document.querySelectorAll('.model-checkbox:checked'))
        .map(cb => cb.value);
    
    if (selectedModels.length === 0) {
        showStatus('Please select at least one model to train.', 'error');
        return;
    }
    
    const options = {
        target_column: targetColumnSelect.value,
        models: selectedModels
    };
    
    // Show progress bar
    progressContainer.style.display = 'block';
    updateProgress(0, 'Initializing model training...');
    
    try {
        const response = await fetch(`${API_BASE_URL}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(options)
        });
        
        // Simulate progress updates
        const progressInterval = setInterval(() => {
            const currentProgress = parseInt(progressFill.style.width) || 0;
            if (currentProgress < 90) {
                updateProgress(currentProgress + 10, 'Training models...');
            }
        }, 500);
        
        const data = await response.json();
        clearInterval(progressInterval);
        
        if (response.ok) {
            updateProgress(100, 'Training completed!');
            state.results = data.results;
            
            // Display results after a short delay
            setTimeout(() => {
                displayResults(data.results, data.problem_type);
                progressContainer.style.display = 'none';
                
                // Show results and download sections
                resultsSection.style.display = 'block';
                downloadSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            }, 1000);
        } else {
            updateProgress(0, `Error: ${data.error}`);
            showStatus(`Training failed: ${data.error}`, 'error');
        }
    } catch (error) {
        updateProgress(0, `Error: ${error.message}`);
        showStatus(`Training failed: ${error.message}`, 'error');
    }
}

function updateProgress(percentage, message) {
    progressFill.style.width = `${percentage}%`;
    progressPercentage.textContent = `${percentage}%`;
    progressMessage.textContent = message;
}

function displayResults(results, problemType) {
    const container = document.getElementById('results-container');
    let html = '';
    
    // Determine metrics based on problem type
    const metrics = problemType === 'classification' ? ['Accuracy'] :
                    problemType === 'regression' ? ['R² Score', 'MSE'] :
                    ['Silhouette Score'];
    
    // Create comparison table
    html += '<h3>Model Performance Comparison</h3>';
    html += '<div style="overflow-x: auto;">';
    html += '<table class="results-table">';
    html += '<tr><th>Model</th><th>Metric</th><th>Original Data</th><th>Processed Data</th><th>Improvement</th></tr>';
    
    // Get all unique model names
    const allModels = new Set([
        ...Object.keys(results.original || {}),
        ...Object.keys(results.processed || {})
    ]);
    
    allModels.forEach(model => {
        const originalResult = results.original?.[model];
        const processedResult = results.processed?.[model];
        
        if (problemType === 'classification') {
            const origAcc = originalResult?.accuracy;
            const procAcc = processedResult?.accuracy;
            
            if (origAcc !== undefined || procAcc !== undefined) {
                const improvement = procAcc !== undefined && origAcc !== undefined ? 
                    ((procAcc - origAcc) * 100).toFixed(2) + '%' : 'N/A';
                
                html += `
                    <tr>
                        <td><strong>${model}</strong></td>
                        <td>Accuracy</td>
                        <td>${formatMetric(origAcc, 'accuracy')}</td>
                        <td>${formatMetric(procAcc, 'accuracy')}</td>
                        <td>${improvement}</td>
                    </tr>
                `;
            }
        } else if (problemType === 'regression') {
            if (originalResult?.r2_score !== undefined || processedResult?.r2_score !== undefined) {
                const origR2 = originalResult?.r2_score;
                const procR2 = processedResult?.r2_score;
                const improvement = procR2 !== undefined && origR2 !== undefined ? 
                    ((procR2 - origR2) * 100).toFixed(2) + '%' : 'N/A';
                
                html += `
                    <tr>
                        <td><strong>${model}</strong></td>
                        <td>R² Score</td>
                        <td>${formatMetric(origR2, 'r2')}</td>
                        <td>${formatMetric(procR2, 'r2')}</td>
                        <td>${improvement}</td>
                    </tr>
                `;
            }
            
            if (originalResult?.mse !== undefined || processedResult?.mse !== undefined) {
                const origMSE = originalResult?.mse;
                const procMSE = processedResult?.mse;
                const improvement = procMSE !== undefined && origMSE !== undefined ? 
                    ((origMSE - procMSE) / origMSE * 100).toFixed(2) + '%' : 'N/A';
                
                html += `
                    <tr>
                        <td><strong>${model}</strong></td>
                        <td>MSE</td>
                        <td>${formatMetric(origMSE, 'mse')}</td>
                        <td>${formatMetric(procMSE, 'mse')}</td>
                        <td>${improvement}</td>
                    </tr>
                `;
            }
        } else if (problemType === 'clustering') {
            const origScore = originalResult?.['silhouette_score'];
            const procScore = processedResult?.['silhouette_score'];
            const improvement = procScore !== undefined && origScore !== undefined ? 
                ((procScore - origScore) * 100).toFixed(2) + '%' : 'N/A';
            
            html += `
                <tr>
                    <td><strong>${model}</strong></td>
                    <td>Silhouette Score</td>
                    <td>${formatMetric(origScore, 'silhouette')}</td>
                    <td>${formatMetric(procScore, 'silhouette')}</td>
                    <td>${improvement}</td>
                </tr>
            `;
        }
    });
    
    html += '</table></div>';
    
    // Add visualization if we have results
    if (Object.keys(results.original || {}).length > 0) {
        html += '<div class="chart-container">';
        html += '<h4>Performance Comparison Chart</h4>';
        html += '<canvas id="resultsChart" width="400" height="200"></canvas>';
        html += '</div>';
    }
    
    container.innerHTML = html;
    
    // Create chart if results exist
    if (Object.keys(results.original || {}).length > 0) {
        createComparisonChart(results, problemType);
    }
}

function formatMetric(value, metricType) {
    if (value === undefined || value === 'Error') return 'N/A';
    
    if (typeof value === 'number') {
        const formatted = value.toFixed(4);
        
        // Add color coding based on metric value
        if (metricType === 'accuracy' || metricType === 'r2' || metricType === 'silhouette') {
            if (value >= 0.8) return `<span class="metric-badge good">${formatted}</span>`;
            if (value >= 0.6) return `<span class="metric-badge medium">${formatted}</span>`;
            return `<span class="metric-badge poor">${formatted}</span>`;
        } else if (metricType === 'mse') {
            if (value < 0.1) return `<span class="metric-badge good">${formatted}</span>`;
            if (value < 1) return `<span class="metric-badge medium">${formatted}</span>`;
            return `<span class="metric-badge poor">${formatted}</span>`;
        }
        
        return formatted;
    }
    
    return value;
}

function createComparisonChart(results, problemType) {
    const ctx = document.getElementById('resultsChart').getContext('2d');
    
    // Prepare data
    const models = Object.keys(results.original || {});
    const originalData = models.map(model => {
        const result = results.original[model];
        if (problemType === 'classification') return result?.accuracy || 0;
        if (problemType === 'regression') return result?.r2_score || 0;
        return result?.['silhouette_score'] || 0;
    });
    
    const processedData = models.map(model => {
        const result = results.processed?.[model];
        if (!result) return null;
        if (problemType === 'classification') return result.accuracy || 0;
        if (problemType === 'regression') return result.r2_score || 0;
        return result['silhouette_score'] || 0;
    });
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [
                {
                    label: 'Original Data',
                    data: originalData,
                    backgroundColor: 'rgba(79, 70, 229, 0.7)',
                    borderColor: 'rgba(79, 70, 229, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Processed Data',
                    data: processedData,
                    backgroundColor: 'rgba(16, 185, 129, 0.7)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: problemType === 'classification' ? 'Accuracy' : 
                              problemType === 'regression' ? 'R² Score' : 'Silhouette Score'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Model Performance Comparison'
                }
            }
        }
    });
}

async function downloadProcessedDataset() {
    if (!state.processedFilename) {
        showStatus('No processed dataset available.', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/download/${state.processedFilename}`);
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `processed_${state.processedFilename}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            showStatus('Download started!', 'success');
        } else {
            const error = await response.json();
            showStatus(`Download failed: ${error.error}`, 'error');
        }
    } catch (error) {
        showStatus(`Download failed: ${error.message}`, 'error');
    }
}

// Health check on load
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('Backend is healthy');
        }
    } catch (error) {
        console.warn('Backend health check failed:', error.message);
    }
}

// Initialize
checkBackendHealth();