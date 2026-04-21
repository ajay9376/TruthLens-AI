// TruthLens AI — Content Script
// Runs on YouTube, Twitter, Facebook, Instagram

(function() {
    'use strict';

    // Add TruthLens badge to videos
    function addBadgeToVideos() {
        const videos = document.querySelectorAll('video:not([truthlens-checked])');

        videos.forEach(video => {
            video.setAttribute('truthlens-checked', 'true');

            // Create badge
            const badge = document.createElement('div');
            badge.style.cssText = `
                position: absolute;
                top: 8px;
                left: 8px;
                background: rgba(124, 58, 237, 0.9);
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                font-family: sans-serif;
                z-index: 9999;
                cursor: pointer;
                backdrop-filter: blur(4px);
            `;
            badge.textContent = '🔍 TruthLens AI';
            badge.title = 'Click to analyze with TruthLens AI';

            badge.addEventListener('click', () => {
                window.open(
                    'https://huggingface.co/spaces/Ajay3323/TruthLens-AI',
                    '_blank'
                );
            });

            // Position badge
            const parent = video.parentElement;
            if (parent) {
                const parentStyle = window.getComputedStyle(parent);
                if (parentStyle.position === 'static') {
                    parent.style.position = 'relative';
                }
                parent.appendChild(badge);
            }
        });
    }

    // Run on page load
    addBadgeToVideos();

    // Watch for new videos
    const observer = new MutationObserver(() => {
        addBadgeToVideos();
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

})();