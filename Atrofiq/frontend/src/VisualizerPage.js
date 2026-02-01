import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Volume2D3DViewer from './components/Volume2D3DViewer';
import './Dashboard.css';

const VisualizerPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { volumeData, analysisResults, studyName } = location.state || {};
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // If no data passed, use demo data instead of redirecting
    if (!volumeData && !analysisResults) {
      console.log('No data passed to VisualizerPage, using demo data');
      // Don't redirect, let the component handle demo mode
    }
  }, [volumeData, analysisResults]);

  const handleGoBack = () => {
    navigate(-1); // Go back to previous page
  };

  const handleGoToDashboard = () => {
    navigate('/dashboard');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <button
                onClick={handleGoBack}
                className="flex items-center px-4 py-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
                </svg>
                Back
              </button>
              
              <div className="h-6 w-px bg-gray-300"></div>
              
              <div>
                <h1 className="text-2xl font-bold text-gray-900 flex items-center">
                  🧠 3D Brain Visualizer
                  {studyName && (
                    <span className="ml-3 text-lg text-gray-500">({studyName})</span>
                  )}
                </h1>
                <p className="text-sm text-gray-600 mt-1">
                  Interactive 3D visualization of brain volumes and analysis results
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={handleGoToDashboard}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Dashboard
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
              <p className="text-gray-600">Loading 3D visualization...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
            <div className="flex items-center">
              <div className="text-red-400 text-xl mr-3">⚠️</div>
              <div>
                <h3 className="text-red-800 font-medium">Visualization Error</h3>
                <p className="text-red-600 text-sm mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {!loading && !error && (
          <div className="bg-white rounded-lg shadow-lg overflow-hidden">
            <Volume2D3DViewer
              isOpen={true}
              onClose={() => {}} // No close needed since it's a full page
              volumeData={volumeData || (analysisResults ? { 
                type: 'nifti_3d', 
                input_file: analysisResults?.metadata?.file_path || 'brain.nii',
                file_path: analysisResults?.metadata?.file_path 
              } : {
                // Demo data when no real data is available
                type: 'demo_2d3d',
                input_file: 'demo_brain_slices.dcm',
                file_path: 'demo_data',
                study_name: studyName || 'Demo Study',
                demo: true
              })}
              isEmbedded={true}
              fullPage={true}
            />
          </div>
        )}

        {/* Info Cards */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h3 className="font-semibold text-gray-900 mb-2 flex items-center">
              🎯 Navigation
            </h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Mouse drag: Rotate brain</li>
              <li>• Mouse wheel: Zoom in/out</li>
              <li>• View controls: Switch modes</li>
            </ul>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h3 className="font-semibold text-gray-900 mb-2 flex items-center">
              📊 Visualization
            </h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• 3D Brain: Interactive volume</li>
              <li>• Brain regions: Color coded</li>
              <li>• Slice view: 2D cross-sections</li>
            </ul>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h3 className="font-semibold text-gray-900 mb-2 flex items-center">
              🔧 Controls
            </h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• View modes: Multiple options</li>
              <li>• Download: Export 3D volumes</li>
              <li>• Share: Copy visualization link</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VisualizerPage;