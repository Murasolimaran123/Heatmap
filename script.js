let video = document.getElementById('videoElement');
let canvas = document.getElementById('canvasOutput');
let ctx = canvas.getContext('2d');
let streaming = false;
let isThermalMode = true;
let cap;
let src;
let dst;
let gray;
let blurred;
let animationId;

// Called when OpenCV.js is fully loaded
function onOpenCvReady() {
    document.getElementById('loadingMessage').innerText = 'Requesting Camera Access...';
    startCamera();
}

// Global error handler for the script
window.onerror = function(message, source, lineno, colno, error) {
    if (message.includes('cv is not defined')) {
        document.getElementById('loadingMessage').innerText = 'Error: Failed to load OpenCV.js';
    }
};

// Request camera access and start stream
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 }, 
                height: { ideal: 480 },
                facingMode: 'user'
            }, 
            audio: false 
        });
        video.srcObject = stream;
        video.play();
    } catch (err) {
        console.error("An error occurred: ", err);
        document.getElementById('loadingMessage').innerText = 'Error: Cannot access camera. Please allow permissions.';
    }
}

// When the video is ready to play
video.addEventListener('canplay', function(ev) {
    if (!streaming) {
        // Match canvas dimensions to actual video dimensions
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        try {
            // Initialize OpenCV standard matrices required for processing
            src = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC4);
            dst = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC4);
            gray = new cv.Mat();
            blurred = new cv.Mat();
            
            // Video capture element
            cap = new cv.VideoCapture(video);
            
            streaming = true;
            document.getElementById('loadingMessage').style.display = 'none';
            
            // Begin the frame processing loop
            requestAnimationFrame(processFrame);
        } catch (e) {
            console.error('OpenCV Initialization Error:', e);
            document.getElementById('loadingMessage').innerText = 'Error: Failed to initialize camera properly.';
        }
    }
}, false);

// Recursive process looping with requestAnimationFrame for 30+ FPS
function processFrame() {
    if (!streaming) return;
    
    try {
        // Read current video frame into the 'src' Mat
        cap.read(src);
        
        if (isThermalMode) {
            generateHeatmap();
        } else {
            // Render normal view by passing Mat to canvas
            cv.imshow('canvasOutput', src);
        }
    } catch (err) {
        console.error("Error processing frame: ", err);
    }
    
    // Request next frame
    animationId = requestAnimationFrame(processFrame);
}

// Applies image processing filters to achieve thermal effect
function generateHeatmap() {
    // 1. Convert frame to grayscale (intensity maps better to heatmap values)
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    
    // 2. Apply Gaussian blur to smooth the image and simulate thermal bleed/diffusion
    let ksize = new cv.Size(15, 15);
    cv.GaussianBlur(gray, blurred, ksize, 0, 0, cv.BORDER_DEFAULT);
    
    // 3. Normalize intensity values (enhances contrast of the thermal map dynamically)
    cv.normalize(blurred, blurred, 0, 255, cv.NORM_MINMAX);
    
    // 4. Apply false-color thermal mapping (Jet perfectly maps Blue->Green->Yellow->Red)
    cv.applyColorMap(blurred, dst, cv.COLORMAP_JET);
    
    // 5. Draw the heatmap result to the canvas
    cv.imshow('canvasOutput', dst);
}

// Button Interaction Logic
document.getElementById('btnNormal').addEventListener('click', toggleThermalMode.bind(null, false));
document.getElementById('btnThermal').addEventListener('click', toggleThermalMode.bind(null, true));

function toggleThermalMode(thermalState) {
    isThermalMode = thermalState;
    if (isThermalMode) {
        document.getElementById('btnThermal').classList.add('active');
        document.getElementById('btnNormal').classList.remove('active');
    } else {
        document.getElementById('btnNormal').classList.add('active');
        document.getElementById('btnThermal').classList.remove('active');
    }
}

// Cleanup OpenCV matrices nicely when navigating away
window.addEventListener('beforeunload', () => {
    if (streaming) {
        cancelAnimationFrame(animationId);
        if (src) src.delete();
        if (dst) dst.delete();
        if (gray) gray.delete();
        if (blurred) blurred.delete();
    }
    const mediaStream = video.srcObject;
    if (mediaStream) {
        const tracks = mediaStream.getTracks();
        tracks.forEach(track => track.stop());
    }
});
