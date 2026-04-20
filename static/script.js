// ─── Drag & Drop ───
const uploadZone = document.getElementById('uploadZone');
const fileInput  = document.getElementById('fileInput');

uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ─── Handle File ───
function handleFile(file) {
    const url = URL.createObjectURL(file);
    document.getElementById('videoPreview').src = url;
    document.getElementById('fileInfo').textContent =
        `📹 ${file.name}  ·  ${(file.size / 1024 / 1024).toFixed(2)} MB`;
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('previewZone').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    window._selectedFile = file;
    window._lastResults = null;
}

// ─── Analyze ───
async function analyzeVideo() {
    const file = window._selectedFile;
    if (!file) return;

    showLoading();

    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await fetch('/analyze-with-report', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        console.log("Results:", data);
        window._lastResults = data;
        hideLoading();
        showResults(data);

    } catch (err) {
        hideLoading();
        alert('❌ Analysis failed! Make sure the backend is running.');
        console.error(err);
    }
}

// ─── Loading ───
function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';

    const steps = ['step1', 'step2', 'step3', 'step4'];
    const statuses = [
        'Running SyncNet lip-sync analysis...',
        'Analyzing face texture patterns...',
        'Detecting blink patterns...',
        'Reading lip movements...'
    ];
    const fill = document.getElementById('loadingFill');
    const status = document.getElementById('loadingStatus');

    let i = 0;
    steps.forEach(s => document.getElementById(s).className = 'step');

    const interval = setInterval(() => {
        if (i > 0) {
            document.getElementById(steps[i-1]).className = 'step done';
            document.getElementById(steps[i-1]).textContent =
                '✅ ' + document.getElementById(steps[i-1])
                .textContent.replace('⬡ ', '');
        }
        if (i < steps.length) {
            document.getElementById(steps[i]).className = 'step active';
            status.textContent = statuses[i];
            fill.style.width = ((i + 1) / steps.length * 100) + '%';
            i++;
        } else {
            clearInterval(interval);
            status.textContent = 'Combining signals...';
            fill.style.width = '100%';
        }
    }, 15000);

    window._loadingInterval = interval;
}

function hideLoading() {
    clearInterval(window._loadingInterval);
    document.getElementById('loadingOverlay').style.display = 'none';
}

// ─── Show Results ───
function showResults(data) {
    document.getElementById('resultsSection').style.display = 'block';
    document.getElementById('previewZone').style.display = 'none';

    const verdict = data.verdict;
    const score   = data.final_score;

    // Verdict card
    const colors = {
        'REAL':      '#10b981',
        'SUSPICIOUS':'#f59e0b',
        'DEEPFAKE':  '#ef4444'
    };
    const icons = {
        'REAL': '✅', 'SUSPICIOUS': '⚠️', 'DEEPFAKE': '❌'
    };

    const card = document.getElementById('verdictCard');
    card.style.borderColor = colors[verdict] || '#6b7280';
    card.style.background  = `${colors[verdict]}11`;

    document.getElementById('verdictIcon').textContent  = icons[verdict] || '?';
    document.getElementById('verdictLabel').textContent = verdict;
    document.getElementById('verdictLabel').style.color = colors[verdict];
    document.getElementById('verdictScore').textContent =
        `Combined Score: ${score.toFixed(1)}/100`;

    // Signal bars
    setSignal('syncnet', data.syncnet_score);
    setSignal('texture', data.texture_score);
    setSignal('blink',   data.blink_score);
    setSignal('lip',     data.lip_score);
    setSignal('voice',   data.voice_score);

    // ─── Download Report Button ───
    const existingBtn = document.getElementById('downloadBtn');
    if (existingBtn) existingBtn.remove();

    const downloadDiv = document.createElement('div');
    downloadDiv.id = 'downloadBtn';
    downloadDiv.style.textAlign = 'center';
    downloadDiv.style.marginTop = '16px';

    if (data.report_path) {
        // Report generated successfully
        downloadDiv.innerHTML = `
            <a href="/download-report?path=${encodeURIComponent(data.report_path)}"
               style="display:inline-flex; align-items:center; gap:8px;
                      background:linear-gradient(135deg,#7c3aed,#06b6d4);
                      color:white; padding:14px 32px; border-radius:12px;
                      text-decoration:none; font-family:'Syne',sans-serif;
                      font-size:16px; font-weight:700; margin-top:8px;"
               download>
                📄 Download Forensic Report
            </a>
        `;
    } else {
        // Show button anyway — generate on click
        downloadDiv.innerHTML = `
            <button onclick="requestReport()"
               style="display:inline-flex; align-items:center; gap:8px;
                      background:linear-gradient(135deg,#7c3aed,#06b6d4);
                      color:white; padding:14px 32px; border-radius:12px;
                      border:none; cursor:pointer; font-family:'Syne',sans-serif;
                      font-size:16px; font-weight:700; margin-top:8px;">
                📄 Download Forensic Report
            </button>
        `;
    }

    document.getElementById('resultsSection').appendChild(downloadDiv);

    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({
        behavior: 'smooth'
    });
}

// ─── Request Report ───
async function requestReport() {
    const file = window._selectedFile;
    const results = window._lastResults;
    if (!file || !results) return;

    const btn = document.querySelector('#downloadBtn button');
    if (btn) btn.textContent = '⏳ Generating report...';

    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await fetch('/analyze-with-report', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.report_path) {
            window.location.href = `/download-report?path=${encodeURIComponent(data.report_path)}`;
        }
    } catch (err) {
        console.error(err);
        alert('❌ Report generation failed!');
    }
}

function setSignal(name, score) {
    const color = score >= 60 ? '#10b981' : score >= 40 ? '#f59e0b' : '#ef4444';
    document.getElementById(`score-${name}`).textContent = score.toFixed(1);
    document.getElementById(`score-${name}`).style.color = color;
    setTimeout(() => {
        document.getElementById(`fill-${name}`).style.width  = score + '%';
        document.getElementById(`fill-${name}`).style.background = color;
    }, 300);
}

// ─── Reset ───
function resetUI() {
    document.getElementById('uploadZone').style.display = 'block';
    document.getElementById('previewZone').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    const downloadBtn = document.getElementById('downloadBtn');
    if (downloadBtn) downloadBtn.remove();
    fileInput.value = '';
    window._selectedFile = null;
    window._lastResults = null;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}