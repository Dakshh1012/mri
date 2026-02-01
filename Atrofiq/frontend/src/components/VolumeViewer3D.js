import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { Volume } from 'three/examples/jsm/misc/Volume';
import { VolumeRenderShader1 } from 'three/examples/jsm/shaders/VolumeShader';

const VolumeViewer3D = ({ volumeData, isOpen, onClose }) => {
  const mountRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const animationIdRef = useRef(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sliceIndex, setSliceIndex] = useState(0);
  const [maxSlices, setMaxSlices] = useState(0);
  const [viewMode, setViewMode] = useState('3d'); // '3d', 'axial', 'sagittal', 'coronal'
  const [opacity, setOpacity] = useState(0.8);
  const [contrast, setContrast] = useState(1.0);

  useEffect(() => {
    if (isOpen && mountRef.current && volumeData) {
      initThreeJS();
      loadVolumeData();
    }

    return () => {
      cleanup();
    };
  }, [isOpen, volumeData]);

  const initThreeJS = () => {
    const width = window.innerWidth * 0.8;
    const height = window.innerHeight * 0.8;

    // Scene
    sceneRef.current = new THREE.Scene();
    sceneRef.current.background = new THREE.Color(0x000000);

    // Camera
    cameraRef.current = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    cameraRef.current.position.set(0, 0, 200);

    // Renderer
    rendererRef.current = new THREE.WebGLRenderer({ antialias: true });
    rendererRef.current.setSize(width, height);
    rendererRef.current.setPixelRatio(window.devicePixelRatio);
    
    if (mountRef.current) {
      mountRef.current.appendChild(rendererRef.current.domElement);
    }

    // Controls
    controlsRef.current = new OrbitControls(cameraRef.current, rendererRef.current.domElement);
    controlsRef.current.enableDamping = true;
    controlsRef.current.dampingFactor = 0.05;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    sceneRef.current.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    sceneRef.current.add(directionalLight);

    startAnimation();
  };

  const loadVolumeData = async () => {
    try {
      setLoading(true);
      setError(null);

      // If volumeData is a URL to a NIfTI file, we would need to load it
      // For now, we'll create a sample volume visualization
      if (typeof volumeData === 'string') {
        await loadNIfTIFile(volumeData);
      } else {
        await createVolumeVisualization(volumeData);
      }

      setLoading(false);
    } catch (err) {
      console.error('Error loading volume data:', err);
      setError('Failed to load volume data: ' + err.message);
      setLoading(false);
    }
  };

  const loadNIfTIFile = async (fileUrl) => {
    // In a real implementation, you would use a NIfTI.js library
    // For now, we'll create a mock volume
    const size = 128;
    const data = new Uint8Array(size * size * size);
    
    // Create a simple brain-like structure
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        for (let k = 0; k < size; k++) {
          const x = (i - size/2) / (size/2);
          const y = (j - size/2) / (size/2);
          const z = (k - size/2) / (size/2);
          
          const r = Math.sqrt(x*x + y*y + z*z);
          if (r < 0.8) {
            const noise = Math.random() * 0.3;
            const value = Math.max(0, 255 * (0.8 - r + noise));
            data[i + j * size + k * size * size] = value;
          }
        }
      }
    }

    createVolumeFromData(data, size, size, size);
    setMaxSlices(size);
  };

  const createVolumeVisualization = (data) => {
    // Handle different data formats
    if (data.slices) {
      createSliceVisualization(data.slices);
    } else {
      // Create default visualization
      loadNIfTIFile(null);
    }
  };

  const createSliceVisualization = (slices) => {
    if (!slices || slices.length === 0) {
      throw new Error('No slices provided');
    }

    setMaxSlices(slices.length);
    
    // Create texture for current slice
    updateSliceTexture(0);
  };

  const updateSliceTexture = (index) => {
    if (!volumeData?.slices || index >= volumeData.slices.length) return;

    // Remove previous slice geometry
    const objectsToRemove = [];
    sceneRef.current.traverse((child) => {
      if (child.name === 'sliceGeometry') {
        objectsToRemove.push(child);
      }
    });
    objectsToRemove.forEach(obj => sceneRef.current.remove(obj));

    // Create new slice
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 256;

    // Draw slice data (assuming grayscale image data)
    const slice = volumeData.slices[index];
    if (slice instanceof ImageData) {
      ctx.putImageData(slice, 0, 0);
    } else if (typeof slice === 'string') {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        createSliceGeometry(canvas);
      };
      img.src = slice;
      return;
    }

    createSliceGeometry(canvas);
  };

  const createSliceGeometry = (canvas) => {
    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.minFilter = THREE.LinearFilter;

    const geometry = new THREE.PlaneGeometry(100, 100);
    const material = new THREE.MeshBasicMaterial({ 
      map: texture,
      transparent: true,
      opacity: opacity,
      side: THREE.DoubleSide
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'sliceGeometry';
    
    // Position based on view mode
    if (viewMode === 'axial') {
      mesh.rotation.x = 0;
    } else if (viewMode === 'sagittal') {
      mesh.rotation.y = Math.PI / 2;
    } else if (viewMode === 'coronal') {
      mesh.rotation.x = Math.PI / 2;
    }

    sceneRef.current.add(mesh);
  };

  const createVolumeFromData = (data, width, height, depth) => {
    // Create volume rendering (simplified version)
    const size = Math.max(width, height, depth);
    const geometry = new THREE.BoxGeometry(size, size, size);
    
    // Create a simple volume material
    const material = new THREE.MeshLambertMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: opacity,
      wireframe: false
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'volumeGeometry';
    sceneRef.current.add(mesh);
  };

  const startAnimation = () => {
    const animate = () => {
      if (controlsRef.current) {
        controlsRef.current.update();
      }
      
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
      
      animationIdRef.current = requestAnimationFrame(animate);
    };
    animate();
  };

  const cleanup = () => {
    if (animationIdRef.current) {
      cancelAnimationFrame(animationIdRef.current);
    }
    
    if (mountRef.current && rendererRef.current) {
      mountRef.current.removeChild(rendererRef.current.domElement);
    }
    
    if (rendererRef.current) {
      rendererRef.current.dispose();
    }
  };

  const handleSliceChange = (newIndex) => {
    setSliceIndex(newIndex);
    if (viewMode !== '3d') {
      updateSliceTexture(newIndex);
    }
  };

  const handleViewModeChange = (mode) => {
    setViewMode(mode);
    if (mode === '3d') {
      // Show volume rendering
      loadVolumeData();
    } else {
      // Show slice view
      updateSliceTexture(sliceIndex);
    }
  };

  const handleOpacityChange = (value) => {
    setOpacity(value);
    // Update material opacity
    if (sceneRef.current) {
      sceneRef.current.traverse((child) => {
        if (child.material) {
          child.material.opacity = value;
        }
      });
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-90 flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-lg shadow-xl max-w-6xl w-full max-h-screen overflow-hidden">
        {/* Header */}
        <div className="bg-gray-800 px-6 py-4 border-b border-gray-700 flex justify-between items-center">
          <h2 className="text-xl font-semibold text-white">3D Volume Viewer</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex">
          {/* Controls Panel */}
          <div className="bg-gray-800 w-64 p-4 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">View Mode</label>
              <select
                value={viewMode}
                onChange={(e) => handleViewModeChange(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              >
                <option value="3d">3D Volume</option>
                <option value="axial">Axial</option>
                <option value="sagittal">Sagittal</option>
                <option value="coronal">Coronal</option>
              </select>
            </div>

            {viewMode !== '3d' && (
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Slice: {sliceIndex + 1} / {maxSlices}
                </label>
                <input
                  type="range"
                  min="0"
                  max={maxSlices - 1}
                  value={sliceIndex}
                  onChange={(e) => handleSliceChange(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Opacity: {Math.round(opacity * 100)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={opacity}
                onChange={(e) => handleOpacityChange(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Contrast: {Math.round(contrast * 100)}%
              </label>
              <input
                type="range"
                min="0.1"
                max="2"
                step="0.1"
                value={contrast}
                onChange={(e) => setContrast(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div className="pt-4 border-t border-gray-700">
              <p className="text-xs text-gray-400">
                • Mouse: Rotate view<br/>
                • Scroll: Zoom in/out<br/>
                • Right-click + drag: Pan
              </p>
            </div>
          </div>

          {/* Viewer */}
          <div className="flex-1 relative">
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
                <div className="text-white text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p>Loading volume data...</p>
                </div>
              </div>
            )}

            {error && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
                <div className="text-red-400 text-center">
                  <p className="text-lg mb-2">⚠️ Error</p>
                  <p>{error}</p>
                </div>
              </div>
            )}

            <div 
              ref={mountRef} 
              className="w-full h-96 bg-black"
              style={{ minHeight: '600px' }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default VolumeViewer3D;