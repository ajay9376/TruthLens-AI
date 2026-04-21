// TruthLens AI — Browser Extension Popup

const TRUTHLENS_URL = 'https://huggingface.co/spaces/Ajay3323/TruthLens-AI';
const LOCAL_URL     = 'http://localhost:8000';

// Open TruthLens AI website
function openTruthLens() {
    chrome.tabs.create({ url: TRUTHLENS_URL });
}

// Analyze current page
async function analyzeCurrentPage() {
    const btn    = document.getElementById('analyzeBtn');
    const icon   = document.getElementById('statusIcon');
    const text   = document.getElementById('statusText');
    const sub    = document.getElementById('statusSub');
    const pBar   = document.getElementById('progressBar');
    const pFill  = document.getElementById('progressFill');

    // Show loading
    btn.disabled  = true;
    icon.innerHTML = '<span class="loading">⬡</span>';
    text.textContent = 'Finding video...';
    sub.textContent  = 'Looking for video on current page';
    pBar.style.display = 'block';
    pFill.style.width  = '10%';

    try {
        // Get current tab
        const [tab] = await chrome.tabs.query({
            active: true, currentWindow: true
        });

        // Inject content script to find video URL
        const results = await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: findVideoUrl,
        });

        const videoUrl = results[0]?.result;

        if (!videoUrl) {
            showError('No video found on this page!');
            btn.disabled = false;
            return;
        }

        text.textContent = 'Video found!';
        sub.textContent  = 'Sending to TruthLens AI...';
        pFill.style.width = '30%';

        // Show message to open TruthLens
        pFill.style.width = '100%';
        icon.textContent  = '🎯';
        text.textContent  = 'Video Detected!';
        sub.textContent   = 'Open TruthLens AI to analyze this video';

        // Store video URL
        chrome.storage.local.set({ detectedVideoUrl: videoUrl });

        // Add analyze button
        document.getElementById('analyzeBtn').textContent = '🚀 Open & Analyze';
        document.getElementById('analyzeBtn').onclick = () => {
            chrome.tabs.create({ url: TRUTHLENS_URL });
        };
        btn.disabled = false;

    } catch (err) {
        showError('Error: ' + err.message);
        btn.disabled = false;
    }
}

// Find video URL on page
function findVideoUrl() {
    // Check for YouTube
    if (window.location.hostname.includes('youtube.com')) {
        const videoId = new URLSearchParams(window.location.search).get('v');
        if (videoId) return `https://www.youtube.com/watch?v=${videoId}`;
    }

    // Check for video elements
    const videos = document.querySelectorAll('video');
    for (const video of videos) {
        if (video.src) return video.src;
        const source = video.querySelector('source');
        if (source?.src) return source.src;
    }

    return null;
}

function showError(msg) {
    document.getElementById('statusIcon').textContent = '❌';
    document.getElementById('statusText').textContent = 'Error';
    document.getElementById('statusSub').textContent  = msg;
    document.getElementById('progressBar').style.display = 'none';
}

// Load saved results
chrome.storage.local.get(['lastResults'], (data) => {
    if (data.lastResults) {
        showResults(data.lastResults);
    }
});

function showResults(results) {
    const verdict = results.verdict;
    const score   = results.final_score;

    const colors = {
        'REAL': '#10b981',
        'DEEPFAKE': '#ef4444',
        'SUSPICIOUS': '#f59e0b'
    };

    const icons = {
        'REAL': '✅', 'DEEPFAKE': '❌', 'SUSPICIOUS': '⚠️'
    };

    document.getElementById('statusIcon').textContent  = icons[verdict];
    document.getElementById('statusText').textContent  = verdict;
    document.getElementById('statusText').style.color  = colors[verdict];
    document.getElementById('statusSub').textContent   = `Score: ${score.toFixed(1)}/100`;
    document.getElementById('signalsDiv').style.display = 'block';

    const setSignal = (id, score) => {
        const el = document.getElementById(`sig-${id}`);
        el.textContent = `${score.toFixed(1)}/100`;
        el.className = `signal-score ${score >= 60 ? 'real' : score >= 40 ? 'sus' : 'fake'}`;
    };

    setSignal('syncnet', results.syncnet_score);
    setSignal('texture', results.texture_score);
    setSignal('blink',   results.blink_score);
    setSignal('lip',     results.lip_score);
    setSignal('voice',   results.voice_score);
}