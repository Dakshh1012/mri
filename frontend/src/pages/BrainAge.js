import React, { useState } from 'react';
import axios from 'axios';

const BrainAge = () => {
  const [file, setFile] = useState(null);
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [jsonFile, setJsonFile] = useState(null);
  const [uploadMode, setUploadMode] = useState('individual'); // 'individual' or 'json'
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file || !age || !gender) {
      setError('Please fill in all fields');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('nifti_file', file);
    formData.append('age', age);
    formData.append('gender', gender);

    try {
      const response = await axios.post('http://localhost:8000/brain-age', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 180000, // 3 minutes timeout
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while processing your request');
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const fileName = selectedFile.name.toLowerCase();
      if (fileName.endsWith('.nii') || fileName.endsWith('.nii.gz')) {
        setFile(selectedFile);
        setError(null);
      } else {
        setError('Please select a valid NIfTI file (.nii or .nii.gz)');
        setFile(null);
      }
    }
  };

  const handleJsonFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const fileName = selectedFile.name.toLowerCase();
      if (fileName.endsWith('.json')) {
        setJsonFile(selectedFile);
        setError(null);
        
        // Read and validate JSON file
        const reader = new FileReader();
        reader.onload = (event) => {
          try {
            const jsonData = JSON.parse(event.target.result);
            if (jsonData.age && jsonData.gender) {
              setAge(jsonData.age.toString());
              setGender(jsonData.gender);
            } else {
              setError('JSON file must contain "age" and "gender" fields');
              setJsonFile(null);
            }
          } catch (err) {
            setError('Invalid JSON file format');
            setJsonFile(null);
          }
        };
        reader.readAsText(selectedFile);
      } else {
        setError('Please select a valid JSON file (.json)');
        setJsonFile(null);
      }
    }
  };

  return (
    <div className="card">
      <h1>🧠 Brain Age Prediction</h1>
      <p>Upload your MRI scan to predict brain age and calculate the brain age gap.</p>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="file">MRI File (NIfTI format):</label>
          <input
            type="file"
            id="file"
            accept=".nii,.nii.gz"
            onChange={handleFileChange}
            required
          />
          {file && (
            <small style={{ color: '#28a745', marginTop: '5px', display: 'block' }}>
              ✓ Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </small>
          )}
        </div>

        <div className="form-group">
          <label htmlFor="age">Age (years):</label>
          <input
            type="number"
            id="age"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            min="18"
            max="100"
            step="0.1"
            placeholder="Enter age in years"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="gender">Gender:</label>
          <select
            id="gender"
            value={gender}
            onChange={(e) => setGender(e.target.value)}
            required
          >
            <option value="">Select Gender</option>
            <option value="M">Male</option>
            <option value="F">Female</option>
          </select>
        </div>

        {/* JSON Upload Button */}
        <div className="form-group">
          <label htmlFor="jsonFile">Or load from JSON file:</label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <input
              type="file"
              id="jsonFile"
              accept=".json"
              onChange={handleJsonFileChange}
              style={{ display: 'none' }}
            />
            <button
              type="button"
              onClick={() => document.getElementById('jsonFile').click()}
              style={{
                padding: '8px 16px',
                backgroundColor: '#2196f3',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px'
              }}
            >
              � Load JSON Metadata
            </button>
            {jsonFile && (
              <small style={{ color: '#28a745' }}>
                ✓ Loaded: {jsonFile.name}
              </small>
            )}
          </div>
          <small style={{ color: '#666', marginTop: '5px', display: 'block' }}>
            JSON will auto-fill age and gender fields above
          </small>
        </div>

        <button type="submit" className="btn" disabled={loading}>
          {loading && <span className="loading-spinner"></span>}
          {loading ? 'Processing...' : 'Predict Brain Age'}
        </button>
      </form>

      {error && (
        <div className="results error">
          <h3>❌ Error</h3>
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className="results success">
          <h3>✅ Brain Age Prediction Results</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px', marginTop: '20px' }}>
            
            <div style={{ background: '#f8f9fa', padding: '15px', borderRadius: '8px' }}>
              <h4>📊 Basic Information</h4>
              <p><strong>Participant ID:</strong> {result.participant_id}</p>
              <p><strong>Status:</strong> {result.status}</p>
              <p><strong>Processing Time:</strong> {result.processing_time_seconds}s</p>
            </div>

            <div style={{ background: '#e3f2fd', padding: '15px', borderRadius: '8px' }}>
              <h4>🧠 Age Analysis</h4>
              <p><strong>Chronological Age:</strong> {result.chronological_age} years</p>
              <p><strong>Predicted Brain Age:</strong> {result.predicted_brain_age.toFixed(2)} years</p>
              <p style={{ 
                color: result.brain_age_gap > 0 ? '#d32f2f' : '#388e3c',
                fontWeight: 'bold'
              }}>
                <strong>Brain Age Gap:</strong> {result.brain_age_gap.toFixed(2)} years
                {result.brain_age_gap > 0 ? ' (Older than expected)' : ' (Younger than expected)'}
              </p>
            </div>

            <div style={{ background: '#f3e5f5', padding: '15px', borderRadius: '8px' }}>
              <h4>🧬 Brain Volume Analysis</h4>
              <p><strong>Total Brain Volume:</strong> {(result.volumetric_features.total_brain / 1000000).toFixed(1)} cm³</p>
              <p><strong>Gray Matter:</strong> {(result.volumetric_features.gray_matter / 1000000).toFixed(1)} cm³</p>
              <p><strong>White Matter:</strong> {(result.volumetric_features.white_matter / 1000000).toFixed(1)} cm³</p>
              <p><strong>CSF:</strong> {(result.volumetric_features.csf / 1000000).toFixed(1)} cm³</p>
              <p><strong>Regions Analyzed:</strong> {Object.keys(result.volumetric_features).length}</p>
            </div>

          </div>

          <details style={{ marginTop: '20px' }}>
            <summary style={{ cursor: 'pointer', fontWeight: 'bold', padding: '10px 0' }}>
              📋 View Detailed Results
            </summary>
            <pre style={{ 
              background: '#f1f3f4', 
              padding: '15px', 
              borderRadius: '5px', 
              overflow: 'auto',
              fontSize: '12px'
            }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
};

export default BrainAge;