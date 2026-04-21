// TruthLens AI — Background Service Worker

chrome.runtime.onInstalled.addListener(() => {
    console.log('TruthLens AI Extension installed!');
});

// Listen for messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'VIDEO_FOUND') {
        chrome.storage.local.set({ detectedVideoUrl: message.url });
        sendResponse({ success: true });
    }
});