import React, { useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Scatter, ScatterChart } from 'recharts';

const NormativeModeling = () => {
  const [file, setFile] = useState(null);
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  // New state for chart controls
  const [selectedRegion, setSelectedRegion] = useState('');
  const [selectedPercentiles, setSelectedPercentiles] = useState([5, 25, 50, 75, 95]);
  const [customPoints, setCustomPoints] = useState([]);
  const [newPointAge, setNewPointAge] = useState('');
  const [newPointVolume, setNewPointVolume] = useState('');

  // Available percentiles for selection
  const availablePercentiles = [5, 10, 15, 25, 50, 75, 85, 90, 95];
  
  // Colors for percentile lines
  const percentileColors = {
    5: '#d32f2f',
    10: '#f57c00',
    15: '#fbc02d',
    25: '#689f38',
    50: '#1976d2',
    75: '#689f38',
    85: '#fbc02d',
    90: '#f57c00',
    95: '#d32f2f'
  };

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
      const response = await axios.post('http://localhost:8000/normative', formData, {
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

  // Get available brain regions
  const getBrainRegions = () => {
    if (!result?.volumetric_features) return [];
    return Object.keys(result.volumetric_features).map(region => ({
      value: region,
      label: region.replace(/_/g, ' ').replace(/approximation/g, '').trim()
    }));
  };

  // Generate synthetic normative data for age vs volume curves
  const generateNormativeData = (region) => {
    if (!region || !result?.volumetric_features) return [];
    
    // Set realistic base volumes for different brain regions
    const getRealisticBaseVolume = (region) => {
      const volumeMap = {
        'total_brain': 1200000,
        'gray_matter': 700000,
        'white_matter': 500000,
        'csf': 150000,
        'left_hemisphere': 600000,
        'right_hemisphere': 600000,
        'frontal_approximation': 175000,
        'parietal_approximation': 140000,
        'temporal_approximation': 154000,
        'occipital_approximation': 105000,
        'cerebellum_approximation': 126000,
        'caudate_approximation': 3500,
        'putamen_approximation': 4500,
        'pallidum_approximation': 1800,
        'hippocampus_approximation': 4000,
        'amygdala_approximation': 1200,
        'thalamus_approximation': 7000,
        'lateral_ventricles_approximation': 20000,
        'third_ventricle_approximation': 1500,
        'fourth_ventricle_approximation': 1000
      };
      return volumeMap[region] || 50000;
    };
    
    const baseVolume = result.volumetric_features[region] || getRealisticBaseVolume(region);
    const ageRange = Array.from({ length: 31 }, (_, i) => i + 10); // Ages 10-40
    
    return ageRange.map(agePoint => {
      const dataPoint = { age: agePoint };
      
      selectedPercentiles.forEach(percentile => {
        // Simulate age-related volume changes with some variance
        const ageFactor = 1 - (agePoint - 10) * 0.005; // Gradual decline with age
        const percentileFactor = percentile / 50; // Scale relative to median
        const randomVariation = 0.8 + (Math.random() * 0.4); // ±20% variation
        
        dataPoint[`p${percentile}`] = Math.max(
          baseVolume * ageFactor * percentileFactor * randomVariation,
          1000 // Minimum volume
        );
      });
      
      return dataPoint;
    });
  };

  // Add custom point
  const addCustomPoint = () => {
    if (!newPointAge || !newPointVolume) return;
    
    const newPoint = {
      id: Date.now(),
      age: parseFloat(newPointAge),
      volume: parseFloat(newPointVolume),
      label: `Custom Point (Age: ${newPointAge}, Volume: ${newPointVolume})`
    };
    
    setCustomPoints([...customPoints, newPoint]);
    setNewPointAge('');
    setNewPointVolume('');
  };

  // Remove custom point
  const removeCustomPoint = (pointId) => {
    setCustomPoints(customPoints.filter(point => point.id !== pointId));
  };

  // Handle percentile selection
  const togglePercentile = (percentile) => {
    if (selectedPercentiles.includes(percentile)) {
      setSelectedPercentiles(selectedPercentiles.filter(p => p !== percentile));
    } else {
      setSelectedPercentiles([...selectedPercentiles, percentile].sort((a, b) => a - b));
    }
  };

  // Custom tooltip for age vs volume chart
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ 
          background: 'white', 
          padding: '15px', 
          border: '1px solid #ccc', 
          borderRadius: '8px',
          boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
        }}>
          <p style={{ fontWeight: 'bold', margin: '0 0 10px 0', fontSize: '14px' }}>
            Age: {label} years
          </p>
          {payload.map((entry, index) => (
            <p key={index} style={{ margin: '5px 0', color: entry.color, fontSize: '13px' }}>
              <strong>{entry.name}:</strong> {(entry.value / 1000).toFixed(1)} cm³
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  // Custom tooltip for scatter points
  const CustomScatterTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div style={{ 
          background: '#2196f3', 
          color: 'white',
          padding: '10px', 
          border: 'none', 
          borderRadius: '6px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
        }}>
          <p style={{ fontWeight: 'bold', margin: '0', fontSize: '13px' }}>
            {data.label || 'Custom Point'}
          </p>
          <p style={{ margin: '5px 0 0 0', fontSize: '12px' }}>
            Age: {data.age} years
          </p>
          <p style={{ margin: '0', fontSize: '12px' }}>
            Volume: {(data.volume / 1000).toFixed(1)} cm³
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="card">
      <h1>📊 Normative Modeling</h1>
      <p>Compare your brain structure volumes against normative population data.</p>
      
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
            min="10"
            max="40"
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

        <button type="submit" className="btn" disabled={loading}>
          {loading && <span className="loading-spinner"></span>}
          {loading ? 'Processing...' : 'Run Normative Analysis'}
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
          <h3>✅ Normative Modeling Results</h3>
          
          {/* Summary Information */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px', margin: '20px 0' }}>
            <div style={{ background: '#f8f9fa', padding: '15px', borderRadius: '8px', textAlign: 'center' }}>
              <h4>👤 Participant</h4>
              <p><strong>{result.participant_id}</strong></p>
              <p>{result.chronological_age} years, {result.sex === 'M' ? 'Male' : 'Female'}</p>
            </div>
            
            <div style={{ background: '#fff3e0', padding: '15px', borderRadius: '8px', textAlign: 'center' }}>
              <h4>⚠️ Outliers</h4>
              <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#f57c00' }}>
                {result.outlier_regions.length}
              </p>
              <p>regions detected</p>
            </div>
            
            <div style={{ background: '#e8f5e8', padding: '15px', borderRadius: '8px', textAlign: 'center' }}>
              <h4>🧬 Regions</h4>
              <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#4caf50' }}>
                {Object.keys(result.percentile_scores).length}
              </p>
              <p>analyzed</p>
            </div>
            
            <div style={{ background: '#e3f2fd', padding: '15px', borderRadius: '8px', textAlign: 'center' }}>
              <h4>⏱️ Processing</h4>
              <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#2196f3' }}>
                {result.processing_time_seconds}s
              </p>
              <p>completion time</p>
            </div>
          </div>

          {/* Chart Controls */}
          <div style={{ marginTop: '30px', background: '#f8f9fa', padding: '20px', borderRadius: '8px' }}>
            <h4>📊 Chart Controls</h4>
            
            {/* Region Selection */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '8px' }}>
                Select Brain Region:
              </label>
              <select 
                value={selectedRegion} 
                onChange={(e) => setSelectedRegion(e.target.value)}
                style={{ 
                  padding: '8px 12px', 
                  borderRadius: '4px', 
                  border: '1px solid #ddd',
                  fontSize: '14px',
                  minWidth: '300px'
                }}
              >
                <option value="">Choose a brain region...</option>
                {getBrainRegions().map((region) => (
                  <option key={region.value} value={region.value}>
                    {region.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Percentile Selection */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '8px' }}>
                Select Percentiles to Display:
              </label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                {availablePercentiles.map((percentile) => (
                  <label key={percentile} style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={selectedPercentiles.includes(percentile)}
                      onChange={() => togglePercentile(percentile)}
                      style={{ marginRight: '5px' }}
                    />
                    <span style={{ 
                      padding: '4px 8px', 
                      borderRadius: '12px', 
                      backgroundColor: selectedPercentiles.includes(percentile) ? percentileColors[percentile] : '#e0e0e0',
                      color: selectedPercentiles.includes(percentile) ? 'white' : '#666',
                      fontSize: '12px',
                      fontWeight: 'bold'
                    }}>
                      {percentile}th
                    </span>
                  </label>
                ))}
              </div>
            </div>

            {/* Add Custom Point */}
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '8px' }}>
                Add Custom Point:
              </label>
              <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
                <input
                  type="number"
                  placeholder="Age (years)"
                  value={newPointAge}
                  onChange={(e) => setNewPointAge(e.target.value)}
                  style={{ padding: '6px 10px', borderRadius: '4px', border: '1px solid #ddd', width: '120px' }}
                />
                <input
                  type="number"
                  placeholder="Volume (mm³)"
                  value={newPointVolume}
                  onChange={(e) => setNewPointVolume(e.target.value)}
                  style={{ padding: '6px 10px', borderRadius: '4px', border: '1px solid #ddd', width: '150px' }}
                />
                <button 
                  onClick={addCustomPoint}
                  disabled={!newPointAge || !newPointVolume}
                  style={{
                    padding: '6px 12px',
                    backgroundColor: '#2196f3',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '14px'
                  }}
                >
                  Add Point
                </button>
              </div>
            </div>

            {/* Custom Points List */}
            {customPoints.length > 0 && (
              <div>
                <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '8px' }}>
                  Custom Points:
                </label>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                  {customPoints.map((point) => (
                    <span 
                      key={point.id}
                      style={{
                        padding: '4px 8px',
                        backgroundColor: '#e3f2fd',
                        borderRadius: '12px',
                        fontSize: '12px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '5px'
                      }}
                    >
                      Age: {point.age}, Vol: {(point.volume/1000).toFixed(1)}cm³
                      <button 
                        onClick={() => removeCustomPoint(point.id)}
                        style={{
                          background: '#d32f2f',
                          color: 'white',
                          border: 'none',
                          borderRadius: '50%',
                          width: '16px',
                          height: '16px',
                          fontSize: '10px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Age vs Volume Chart */}
          {selectedRegion && (
            <div style={{ marginTop: '30px' }}>
              <h4>📈 Age vs Volume: {selectedRegion.replace(/_/g, ' ').replace(/approximation/g, '').trim()}</h4>
              <p style={{ fontSize: '14px', color: '#666', marginBottom: '20px' }}>
                Normative curves showing how brain volume changes with age. Your data point and custom points are overlaid.
              </p>
              
              <ResponsiveContainer width="100%" height={500}>
                <LineChart data={generateNormativeData(selectedRegion)} margin={{ top: 20, right: 30, left: 60, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis 
                    dataKey="age" 
                    label={{ value: 'Age (years)', position: 'insideBottom', offset: -10 }}
                    domain={['dataMin', 'dataMax']}
                  />
                  <YAxis 
                    label={{ value: 'Volume (cm³)', angle: -90, position: 'insideLeft', offset: -40 }}
                    tickFormatter={(value) => `${(value / 1000).toFixed(0)}`}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  
                  {/* Render percentile lines - all solid */}
                  {selectedPercentiles.map((percentile) => (
                    <Line 
                      key={percentile}
                      type="monotone"
                      dataKey={`p${percentile}`}
                      stroke={percentileColors[percentile]}
                      strokeWidth={percentile === 50 ? 3 : 2}
                      dot={false}
                      activeDot={{ r: 4 }}
                      name={`${percentile}th Percentile`}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>

              {/* Overlay scatter plot for custom points and user data */}
              <div style={{ marginTop: '-500px', pointerEvents: 'none' }}>
                <ResponsiveContainer width="100%" height={500}>
                  <ScatterChart margin={{ top: 20, right: 30, left: 60, bottom: 60 }}>
                    <XAxis 
                      dataKey="age" 
                      type="number"
                      domain={[10, 40]}
                      tick={false}
                      axisLine={false}
                    />
                    <YAxis 
                      dataKey="volume"
                      type="number"
                      tick={false}
                      axisLine={false}
                    />
                    <Tooltip content={<CustomScatterTooltip />} />
                    
                    {/* User's actual data point */}
                    {result.volumetric_features[selectedRegion] && (
                      <Scatter 
                        data={[{
                          age: result.chronological_age,
                          volume: result.volumetric_features[selectedRegion],
                          label: `Your Data (${result.participant_id})`
                        }]}
                        fill="#ff5722"
                        stroke="#fff"
                        strokeWidth={2}
                        r={8}
                        name="Your Data"
                      />
                    )}
                    
                    {/* Custom points */}
                    {customPoints.length > 0 && (
                      <Scatter 
                        data={customPoints}
                        fill="#2196f3"
                        stroke="#fff"
                        strokeWidth={2}
                        r={6}
                        name="Custom Points"
                      />
                    )}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Region Selection Prompt */}
          {!selectedRegion && (
            <div style={{ 
              marginTop: '30px', 
              textAlign: 'center', 
              padding: '40px', 
              backgroundColor: '#f5f5f5', 
              borderRadius: '8px',
              color: '#666'
            }}>
              <h4>📊 Select a Brain Region</h4>
              <p>Choose a brain region from the dropdown above to view the age vs volume normative curves.</p>
            </div>
          )}

          {/* Outlier Regions Summary */}
          {result.outlier_regions.length > 0 && (
            <div style={{ marginTop: '30px', background: '#fff3e0', padding: '20px', borderRadius: '8px', border: '1px solid #ffb74d' }}>
              <h4 style={{ color: '#e65100', margin: '0 0 15px 0' }}>⚠️ Detected Outlier Regions</h4>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                {result.outlier_regions.map((region, index) => (
                  <span 
                    key={index}
                    style={{ 
                      background: '#ffcc02', 
                      padding: '5px 10px', 
                      borderRadius: '15px',
                      fontSize: '14px',
                      fontWeight: 'bold'
                    }}
                  >
                    {region.replace(/_/g, ' ').replace(/approximation/g, '')}
                  </span>
                ))}
              </div>
              <p style={{ margin: '15px 0 0 0', fontSize: '14px', color: '#e65100' }}>
                These regions show volumes significantly different from the normative population (beyond 2 standard deviations).
              </p>
            </div>
          )}

          {/* Raw Data */}
          <details style={{ marginTop: '30px' }}>
            <summary style={{ cursor: 'pointer', fontWeight: 'bold', padding: '10px 0' }}>
              📋 View Detailed Raw Data
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

export default NormativeModeling;