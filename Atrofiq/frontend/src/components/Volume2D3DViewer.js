import React, { useState, useEffect, useRef } from 'react';
import { API_BASE_URL } from '../config';

const Volume2D3DViewer = ({ isOpen, onClose, volumeData, isEmbedded = false, fullPage = false }) => {
  const [currentSlice, setCurrentSlice] = useState(0);
  const [maxSlices, setMaxSlices] = useState(0);
  const [viewMode, setViewMode] = useState('comparison');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [conversionResult, setConversionResult] = useState(null);
  const [volumeSlices, setVolumeSlices] = useState([]);
  const [visualizationData, setVisualizationData] = useState(null);
  
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  // Load 2D-3D conversion data
  useEffect(() => {
    if (isOpen && volumeData) {
      load2D3DConversion();
    }
  }, [isOpen, volumeData]);

  const load2D3DConversion = async () => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('Starting 2D-3D conversion with volumeData:', volumeData);
      
      // Call the 2D-3D conversion API
      console.log('Calling API: /convert-2d-3d/test_folder');
      const response = await fetch(`${API_BASE_URL}/convert-2d-3d/test_folder`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          slice_data: volumeData,
          generate_visualization: true 
        })
      });
      
      console.log('API Response status:', response.status, response.statusText);
      
      if (!response.ok) {
        throw new Error(`Conversion failed: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      console.log('2D-3D Conversion API Result:', JSON.stringify(result, null, 2));
      
      // Handle both nested and flat response structures
      const conversionData = result.volume_2d3d || result;
      console.log('Using conversionData:', JSON.stringify(conversionData, null, 2));
      
      if (!conversionData.success) {
        throw new Error(conversionData.error || 'Conversion failed');
      }
      
      setConversionResult(conversionData);
      
      // Set up slice navigation if we have output dimensions
      if (conversionData.output_shape && conversionData.output_shape.length >= 3) {
        const numSlices = conversionData.output_shape[2]; // Z-axis
        console.log('Setting up slice navigation with', numSlices, 'slices');
        setMaxSlices(numSlices);
        setCurrentSlice(Math.floor(numSlices / 2)); // Start from middle slice
      }
      
      // Load visualization file if available
      if (conversionData.visualization_file) {
        console.log('Loading visualization:', conversionData.visualization_file);
        await loadVisualizationImage(conversionData.visualization_file);
      }
      
      // Load volume slices for navigation  
      if (conversionData.output_3d_file) {
        console.log('Loading volume slices for:', conversionData.output_3d_file);
        await loadVolumeSlices(conversionData.output_3d_file);
      }
      
    } catch (err) {
      console.error('2D-3D conversion failed:', err);
      setError(`Conversion failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const loadVisualizationImage = async (visualizationPath) => {
    try {
      // Load the comparison visualization image
      const imageUrl = `${API_BASE_URL}/files/${visualizationPath}`;
      const img = new Image();
      
      return new Promise((resolve, reject) => {
        img.onload = () => {
          setVisualizationData({
            image: img,
            url: imageUrl,
            width: img.width,
            height: img.height
          });
          resolve();
        };
        img.onerror = () => {
          console.warn('Failed to load visualization image, using fallback');
          resolve(); // Don't fail completely
        };
        img.src = imageUrl;
      });
    } catch (err) {
      console.warn('Error loading visualization:', err);
    }
  };

  const loadVolumeSlices = async (volumePath) => {
    try {
      // Load individual slices from the 3D volume
      const response = await fetch(`${API_BASE_URL}/volumes/slice-info/${volumePath}`);
      if (response.ok) {
        const sliceInfo = await response.json();
        setVolumeSlices(sliceInfo.slices || []);
      }
    } catch (err) {
      console.warn('Failed to load volume slices:', err);
    }
  };

  const handleSliceChange = (newSlice) => {
    if (newSlice >= 0 && newSlice < maxSlices) {
      setCurrentSlice(newSlice);
    }
  };

  const renderComparisonView = () => {
    return (
      <div className="comparison-view">
        <h3>2D Input → 3D Output Comparison</h3>
        {visualizationData ? (
          <div className="comparison-container">
            {visualizationData.demo ? (
              <div className="demo-comparison">
                <div style={{ 
                  width: '100%', 
                  height: '300px', 
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  borderRadius: '8px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontSize: '18px',
                  fontWeight: 'bold'
                }}>
                  <div style={{ marginBottom: '10px' }}>🧠 Demo 2D→3D Conversion</div>
                  <div style={{ fontSize: '14px', opacity: '0.8' }}>
                    Sample brain slice reconstruction visualization
                  </div>
                  <div style={{ 
                    marginTop: '20px', 
                    display: 'flex', 
                    gap: '20px',
                    fontSize: '12px'
                  }}>
                    <div>📊 Input: 2D Slices</div>
                    <div>→</div>
                    <div>🎯 Output: 3D Volume</div>
                  </div>
                </div>
              </div>
            ) : (
              <img 
                src={visualizationData.url} 
                alt="2D-3D Comparison" 
                className="comparison-image"
                style={{ maxWidth: '100%', height: 'auto' }}
              />
            )}
            <div className="comparison-info">
              <p><strong>Input Shape:</strong> {conversionResult?.input_shape?.join(' × ') || 'N/A'}</p>
              <p><strong>Output Shape:</strong> {conversionResult?.output_shape?.join(' × ') || 'N/A'}</p>
              <p><strong>Conversion Method:</strong> {conversionResult?.conversion_method || 'GAN-based 2D→3D'}</p>
              {conversionResult?.demo && (
                <p style={{ color: '#ff6b35', fontWeight: 'bold' }}>
                  🎮 Demo Mode - Start backend API for real data
                </p>
              )}
            </div>
          </div>
        ) : (
          <div className="comparison-placeholder">
            <p>No visualization data available</p>
            {conversionResult && (
              <div className="result-info">
                <p><strong>Input:</strong> {conversionResult.input_shape?.join(' × ')} 2D slices</p>
                <p><strong>Output:</strong> {conversionResult.output_shape?.join(' × ')} 3D volume</p>
                <p><strong>Status:</strong> Conversion completed</p>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  const renderSliceView = () => {
    return (
      <div className="slice-view">
        <h3>3D Volume Navigation</h3>
        <div className="slice-controls">
          <button 
            onClick={() => handleSliceChange(currentSlice - 1)}
            disabled={currentSlice === 0}
          >
            Previous Slice
          </button>
          <span className="slice-info">
            Slice {currentSlice + 1} of {maxSlices}
          </span>
          <button 
            onClick={() => handleSliceChange(currentSlice + 1)}
            disabled={currentSlice >= maxSlices - 1}
          >
            Next Slice
          </button>
        </div>
        
        <div className="slice-viewer">
          {conversionResult?.demo ? (
            <div className="demo-slice">
              <div style={{
                width: '300px',
                height: '300px',
                border: '2px solid #ddd',
                borderRadius: '8px',
                background: `radial-gradient(circle at ${50 + Math.sin(currentSlice / 10) * 20}% ${50 + Math.cos(currentSlice / 8) * 15}%, 
                  #e0e0e0 0%, #c0c0c0 30%, #a0a0a0 60%, #808080 100%)`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto',
                position: 'relative',
                overflow: 'hidden'
              }}>
                {/* Create brain-like pattern */}
                <div style={{
                  position: 'absolute',
                  top: '20%',
                  left: '20%',
                  right: '20%',
                  bottom: '20%',
                  background: `radial-gradient(ellipse 80% 60% at 50% 45%, 
                    rgba(160,160,160,0.8) 0%, 
                    rgba(120,120,120,0.6) 40%, 
                    transparent 70%)`,
                  borderRadius: '50%'
                }} />
                {/* Add some internal structures */}
                <div style={{
                  position: 'absolute',
                  top: '40%',
                  left: '45%',
                  width: '10%',
                  height: '20%',
                  background: 'rgba(100,100,100,0.4)',
                  borderRadius: '50%'
                }} />
                <div style={{
                  position: 'absolute',
                  bottom: '10px',
                  right: '10px',
                  fontSize: '12px',
                  color: '#666',
                  background: 'rgba(255,255,255,0.8)',
                  padding: '2px 6px',
                  borderRadius: '4px'
                }}>
                  Demo Slice {currentSlice + 1}
                </div>
              </div>
            </div>
          ) : volumeSlices[currentSlice] ? (
            <img 
              src={`${API_BASE_URL}/volumes/slice/${volumeSlices[currentSlice]}`}
              alt={`Slice ${currentSlice + 1}`}
              className="slice-image"
              style={{ maxWidth: '100%', height: 'auto', border: '1px solid #ddd' }}
            />
          ) : (
            <div className="slice-placeholder">
              <p>Slice {currentSlice + 1}</p>
              <div style={{ 
                width: '300px', 
                height: '300px', 
                border: '2px solid #ddd', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                backgroundColor: '#f5f5f5'
              }}>
                Loading slice...
              </div>
            </div>
          )}
        </div>
        
        <input
          type="range"
          min="0"
          max={maxSlices - 1}
          value={currentSlice}
          onChange={(e) => handleSliceChange(parseInt(e.target.value))}
          className="slice-slider"
          style={{ width: '100%', margin: '20px 0' }}
        />
      </div>
    );
  };

  const render3DView = () => {
    return (
      <div className="volume-3d-view">
        <h3>3D Volume Visualization</h3>
        {conversionResult?.output_3d_file ? (
          <div className="volume-3d-container">
            <canvas 
              ref={canvasRef}
              width="600" 
              height="400"
              style={{ 
                border: '1px solid #ddd', 
                backgroundColor: '#000',
                display: 'block',
                margin: '0 auto'
              }}
            />
            <div className="volume-info">
              <p><strong>3D Volume File:</strong> {conversionResult.output_3d_file}</p>
              <p><strong>Dimensions:</strong> {conversionResult.output_shape?.join(' × ')}</p>
              <p><strong>Format:</strong> NIfTI (.nii)</p>
            </div>
            <div className="3d-controls">
              <p>Interactive 3D visualization of generated volume</p>
              <p>Use mouse to rotate and zoom</p>
            </div>
          </div>
        ) : (
          <div className="volume-3d-placeholder">
            <p>No 3D volume data available</p>
          </div>
        )}
      </div>
    );
  };

  if (!isOpen) return null;

  const containerClass = fullPage 
    ? 'volume-viewer-fullpage' 
    : isEmbedded 
      ? 'volume-viewer-embedded' 
      : 'volume-viewer-modal';

  return (
    <div className={`volume-viewer-container ${containerClass}`}>
      {!fullPage && !isEmbedded && (
        <div className="modal-backdrop" onClick={onClose}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>2D-3D Volume Viewer</h2>
              <button onClick={onClose} className="close-button">×</button>
            </div>
          </div>
        </div>
      )}
      
      <div className={fullPage ? 'fullpage-content' : 'modal-content'}>
        {!isEmbedded && (
          <div className="viewer-header">
            <h2>2D-3D Conversion Results</h2>
            {!fullPage && (
              <button onClick={onClose} className="close-button">×</button>
            )}
          </div>
        )}
        
        <div className="view-mode-tabs">
          <button 
            className={viewMode === 'comparison' ? 'active' : ''}
            onClick={() => setViewMode('comparison')}
          >
            Comparison View
          </button>
          <button 
            className={viewMode === 'slices' ? 'active' : ''}
            onClick={() => setViewMode('slices')}
            disabled={!maxSlices}
          >
            Slice Navigation ({maxSlices} slices)
          </button>
          <button 
            className={viewMode === '3d' ? 'active' : ''}
            onClick={() => setViewMode('3d')}
            disabled={!conversionResult?.output_3d_file}
          >
            3D Volume
          </button>
        </div>
        
        <div className="viewer-content">
          {loading && (
            <div className="loading-state">
              <p>Running 2D-3D conversion...</p>
              <div className="spinner"></div>
            </div>
          )}
          
          {error && (
            <div className="error-state">
              <h3>Conversion Error</h3>
              <p>{error}</p>
              <button onClick={load2D3DConversion}>Retry Conversion</button>
            </div>
          )}
          
          {!loading && !error && conversionResult && (
            <>
              {viewMode === 'comparison' && renderComparisonView()}
              {viewMode === 'slices' && renderSliceView()}
              {viewMode === '3d' && render3DView()}
            </>
          )}
          
          {!loading && !error && !conversionResult && (
            <div className="no-data-state">
              <h3>No Conversion Data</h3>
              <p>Please upload DICOM files and run analysis to see 2D-3D conversion results.</p>
            </div>
          )}
        </div>
      </div>
      
      <style jsx>{`
        .volume-viewer-container {
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .volume-viewer-modal {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          z-index: 1000;
        }
        
        .modal-backdrop {
          background: rgba(0, 0, 0, 0.8);
          width: 100%;
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .modal-content {
          background: white;
          border-radius: 8px;
          max-width: 90%;
          max-height: 90%;
          overflow: auto;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        
        .volume-viewer-fullpage {
          width: 100%;
          min-height: 100vh;
          background: white;
        }
        
        .fullpage-content {
          padding: 20px;
          max-width: 1200px;
          margin: 0 auto;
        }
        
        .viewer-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px;
          border-bottom: 1px solid #e0e0e0;
        }
        
        .view-mode-tabs {
          display: flex;
          border-bottom: 1px solid #e0e0e0;
          padding: 0 20px;
        }
        
        .view-mode-tabs button {
          padding: 12px 24px;
          border: none;
          background: none;
          cursor: pointer;
          font-weight: 500;
          border-bottom: 2px solid transparent;
          transition: all 0.2s;
        }
        
        .view-mode-tabs button:hover:not(:disabled) {
          background: #f5f5f5;
        }
        
        .view-mode-tabs button.active {
          border-bottom-color: #007bff;
          color: #007bff;
        }
        
        .view-mode-tabs button:disabled {
          color: #ccc;
          cursor: not-allowed;
        }
        
        .viewer-content {
          padding: 20px;
          min-height: 400px;
        }
        
        .comparison-view, .slice-view, .volume-3d-view {
          text-align: center;
        }
        
        .comparison-container {
          margin: 20px 0;
        }
        
        .comparison-info {
          margin-top: 20px;
          background: #f8f9fa;
          padding: 15px;
          border-radius: 4px;
          text-align: left;
        }
        
        .slice-controls {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 20px;
          margin: 20px 0;
        }
        
        .slice-controls button {
          padding: 8px 16px;
          border: 1px solid #ddd;
          background: #fff;
          cursor: pointer;
          border-radius: 4px;
        }
        
        .slice-controls button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .slice-viewer {
          margin: 20px 0;
        }
        
        .loading-state, .error-state, .no-data-state {
          text-align: center;
          padding: 40px 20px;
        }
        
        .spinner {
          width: 40px;
          height: 40px;
          border: 4px solid #f3f3f3;
          border-top: 4px solid #007bff;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 20px auto;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .close-button {
          background: none;
          border: none;
          font-size: 24px;
          cursor: pointer;
          padding: 0;
          width: 30px;
          height: 30px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .error-state button {
          padding: 10px 20px;
          background: #007bff;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          margin-top: 10px;
        }
      `}</style>
    </div>
  );
};

export default Volume2D3DViewer;