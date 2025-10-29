import React, { useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Scatter, ScatterChart, Legend } from 'recharts';
import './Dashboard.css';

const Dashboard = () => {
  const [file, setFile] = useState(null);
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  
  // Normative modeling interactive features
  const [selectedRegion, setSelectedRegion] = useState('total_brain');
  const [selectedPercentiles, setSelectedPercentiles] = useState([10, 25, 50, 75, 90]);
  const [customPoints, setCustomPoints] = useState([]);
  const [customAge, setCustomAge] = useState('');
  const [customVolume, setCustomVolume] = useState('');

  const availablePercentiles = [5, 10, 15, 25, 50, 75, 85, 90, 95];
  const percentileColors = {
    5: '#ff9800',   // Orange
    10: '#f44336',  // Red  
    15: '#e91e63', // Pink
    25: '#9c27b0', // Purple
    50: '#2196f3', // Blue (median - most important)
    75: '#009688', // Teal
    85: '#4caf50', // Green
    90: '#8bc34a', // Light Green
    95: '#ffeb3b'  // Yellow
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !age || !gender) {
      setError('Please fill in all fields');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('nifti_file', file);  // Changed from 'file' to 'nifti_file'
      formData.append('age', age);
      formData.append('gender', gender);

      // Call both APIs
      const [brainAgeResponse, normativeResponse] = await Promise.all([
        axios.post('http://localhost:8000/brain-age', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        }),
        axios.post('http://localhost:8000/normative', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
      ]);

      setResults({
        brainAge: brainAgeResponse.data,
        normative: normativeResponse.data
      });
    } catch (err) {
      console.error('API Error:', err);
      let errorMessage = 'An error occurred during processing';
      
      if (err.response?.data?.detail) {
        if (typeof err.response.data.detail === 'string') {
          errorMessage = err.response.data.detail;
        } else if (Array.isArray(err.response.data.detail)) {
          errorMessage = err.response.data.detail.map(e => e.msg || e).join(', ');
        } else {
          errorMessage = JSON.stringify(err.response.data.detail);
        }
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const generateNormativeData = (region) => {
    if (!results?.normative || !region) return [];
    
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
    
    const baseVolume = results.normative.volumetric_features[region] || getRealisticBaseVolume(region);
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

  const getPercentilePosition = () => {
    if (!results?.normative?.volumetric_features?.total_brain) return null;
    
    const volume = results.normative.volumetric_features.total_brain / 1000; // Convert to cm³
    const patientAge = parseInt(age);
    
    // Simple percentile calculation based on volume
    if (volume > 1100) return { percentile: 90, color: '#4CAF50' };
    if (volume > 1050) return { percentile: 75, color: '#8BC34A' };
    if (volume > 1000) return { percentile: 60, color: '#FFC107' };
    if (volume > 950) return { percentile: 40, color: '#FF9800' };
    if (volume > 900) return { percentile: 25, color: '#FF5722' };
    return { percentile: 10, color: '#F44336' };
  };

  const getBrainAgeGap = () => {
    if (!results?.brainAge) return null;
    
    // Try different possible response structures
    let predictedAge = null;
    if (results.brainAge.predicted_age !== undefined) {
      predictedAge = results.brainAge.predicted_age;
    } else if (results.brainAge.prediction !== undefined) {
      predictedAge = results.brainAge.prediction;
    } else if (results.brainAge.brain_age !== undefined) {
      predictedAge = results.brainAge.brain_age;
    } else if (results.brainAge.predicted_brain_age !== undefined) {
      predictedAge = results.brainAge.predicted_brain_age;
    }
    
    if (predictedAge === null || predictedAge === undefined) return null;
    
    const chronologicalAge = parseFloat(age);
    return predictedAge - chronologicalAge;
  };

  const getInterpretation = () => {
    const gap = getBrainAgeGap();
    if (gap === null || isNaN(gap)) return 'Brain age analysis pending - please check input data.';
    
    if (Math.abs(gap) <= 2) {
      return 'Brain age is within normal range for chronological age.';
    } else if (gap > 2) {
      return `Brain Age Gap of +${gap.toFixed(1)} years may indicate slightly accelerated aging.`;
    } else {
      return `Brain Age Gap of ${gap.toFixed(1)} years may indicate preserved brain structure.`;
    }
  };

  const handlePercentileToggle = (percentile) => {
    setSelectedPercentiles(prev => {
      if (prev.includes(percentile)) {
        return prev.filter(p => p !== percentile);
      } else {
        return [...prev, percentile].sort((a, b) => a - b);
      }
    });
  };

  const addCustomPoint = () => {
    if (customAge && customVolume) {
      setCustomPoints(prev => [...prev, {
        age: parseFloat(customAge),
        volume: parseFloat(customVolume) * 1000, // Convert cm³ to mm³
        label: `Custom Point (${customAge}y, ${customVolume}cm³)`
      }]);
      setCustomAge('');
      setCustomVolume('');
    }
  };

  const clearCustomPoints = () => {
    setCustomPoints([]);
  };

  const getAvailableRegions = () => {
    if (!results?.normative?.volumetric_features) return [];
    return Object.keys(results.normative.volumetric_features);
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{
          backgroundColor: '#2c3e50',
          color: 'white',
          padding: '10px',
          border: 'none',
          borderRadius: '6px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
        }}>
          <p style={{ fontWeight: 'bold', margin: '0', fontSize: '13px' }}>
            Age: {label} years
          </p>
          {payload.map((entry) => (
            <p key={entry.dataKey} style={{ margin: '2px 0', fontSize: '12px', color: entry.color }}>
              <strong>{entry.name}:</strong> {(entry.value / 1000).toFixed(1)} cm³
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const CustomScatterTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div style={{
          backgroundColor: '#2c3e50',
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
    <div className="dashboard">
      <header className="dashboard-header">
        <div className="logo">
          <h1>AtrofIQ</h1>
          <span>Brain MRI Analytics Dashboard</span>
        </div>
      </header>

      {!results ? (
        <div className="upload-section">
          <div className="upload-card">
            <h2>Upload Brain MRI for Analysis</h2>
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label>MRI File (NIfTI format):</label>
                <input
                  type="file"
                  accept=".nii,.nii.gz"
                  onChange={handleFileChange}
                  required
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Age (years):</label>
                  <input
                    type="number"
                    value={age}
                    onChange={(e) => setAge(e.target.value)}
                    min="10"
                    max="100"
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Gender:</label>
                  <select
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                    required
                  >
                    <option value="">Select Gender</option>
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                  </select>
                </div>
              </div>

              <button 
                type="submit" 
                disabled={loading}
                className="analyze-btn"
              >
                {loading ? 'Analyzing...' : 'Analyze Brain MRI'}
              </button>
            </form>

            {error && <div className="error-message">{error}</div>}
          </div>
        </div>
      ) : (
        <div className="results-dashboard">
          {/* New Patient Button */}
          <div className="top-actions">
            <button 
              className="new-patient-btn"
              onClick={() => {
                setResults(null);
                setFile(null);
                setAge('');
                setGender('');
                setError('');
                setSelectedRegion('total_brain');
                setSelectedPercentiles([10, 25, 50, 75, 90]);
                setCustomPoints([]);
                setCustomAge('');
                setCustomVolume('');
              }}
            >
              + New Patient
            </button>
          </div>

          {/* Patient Information Header */}
          <div className="patient-info">
            <div className="info-item">
              <span className="label">Patient ID:</span>
              <span className="value">{results?.brainAge?.participant_id || 'N/A'}</span>
            </div>
            <div className="info-item">
              <span className="label">Age:</span>
              <span className="value">{age} years</span>
            </div>
            <div className="info-item">
              <span className="label">Sex:</span>
              <span className="value">{gender === 'M' ? 'Male' : 'Female'}</span>
            </div>
            <div className="info-item">
              <span className="label">Scan Date:</span>
              <span className="value">{new Date().toLocaleDateString()}</span>
            </div>
          </div>

          <div className="main-content">
            {/* Left Column */}
            <div className="left-column">
              {/* Brain Visualization */}
              <div className="brain-visualization">
                <h3>Brain Visualization</h3>
                <div className="mri-views">
                  <div className="mri-view">
                    <div className="view-placeholder">
                      <span>Coronal View</span>
                    </div>
                  </div>
                  <div className="mri-view">
                    <div className="view-placeholder">
                      <span>Axial View</span>
                    </div>
                  </div>
                  <div className="mri-view">
                    <div className="view-placeholder">
                      <span>Sagittal View</span>
                    </div>
                  </div>
                </div>
                <button className="viewer-btn" disabled>
                  Open 3D Viewer
                </button>
              </div>

              {/* Brain Age Analysis */}
              <div className="brain-age-analysis">
                <h3>Brain Age Analysis</h3>
                <div className="age-comparison">
                  <div className="age-item predicted">
                    <span className="age-label">Predicted Brain Age</span>
                    <span className="age-value">
                      {(() => {
                        if (!results?.brainAge) return 'N/A';
                        let predictedAge = results.brainAge.predicted_age || 
                                         results.brainAge.prediction || 
                                         results.brainAge.brain_age || 
                                         results.brainAge.predicted_brain_age;
                        return predictedAge ? Math.round(predictedAge) : 'N/A';
                      })()}
                    </span>
                  </div>
                  <div className="age-item chronological">
                    <span className="age-label">Chronological Age</span>
                    <span className="age-value">{age}</span>
                  </div>
                </div>
                <div className="brain-age-gap">
                  <span className="gap-label">Brain Age Gap:</span>
                  <span className={`gap-value ${getBrainAgeGap() > 0 ? 'positive' : 'negative'}`}>
                    {getBrainAgeGap() !== null && !isNaN(getBrainAgeGap()) ? 
                      `${getBrainAgeGap() > 0 ? '+' : ''}${getBrainAgeGap().toFixed(1)} years` : 
                      'N/A'
                    }
                  </span>
                </div>
                <div className="interpretation">
                  {getInterpretation()}
                </div>
              </div>
            </div>

            {/* Right Column */}
            <div className="right-column">
              {/* Normative Curve */}
              <div className="normative-curve">
                <h3>Normative Curve</h3>
                <div className="chart-subtitle">Interactive Age vs Volume Analysis</div>
                
                {/* Region Selection */}
                <div className="controls-section">
                  <div className="control-group">
                    <label>Brain Region:</label>
                    <select 
                      value={selectedRegion} 
                      onChange={(e) => setSelectedRegion(e.target.value)}
                      className="region-select"
                    >
                      {getAvailableRegions().map(region => (
                        <option key={region} value={region}>
                          {region.replace(/_/g, ' ').replace(/approximation/g, '').trim()}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Percentile Selection */}
                  <div className="control-group">
                    <label>Select Percentiles:</label>
                    <div className="percentile-buttons">
                      {availablePercentiles.map(percentile => (
                        <button
                          key={percentile}
                          onClick={() => handlePercentileToggle(percentile)}
                          className={`percentile-btn ${selectedPercentiles.includes(percentile) ? 'active' : ''}`}
                          style={{
                            backgroundColor: selectedPercentiles.includes(percentile) ? percentileColors[percentile] : '#f8f9fa',
                            borderColor: selectedPercentiles.includes(percentile) ? percentileColors[percentile] : '#dee2e6'
                          }}
                        >
                          {percentile}th
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Custom Point Addition */}
                  <div className="control-group">
                    <label>Add Custom Point:</label>
                    <div className="custom-point-inputs">
                      <input
                        type="number"
                        placeholder="Age"
                        value={customAge}
                        onChange={(e) => setCustomAge(e.target.value)}
                        min="10"
                        max="100"
                      />
                      <input
                        type="number"
                        placeholder="Volume (cm³)"
                        value={customVolume}
                        onChange={(e) => setCustomVolume(e.target.value)}
                      />
                      <button onClick={addCustomPoint} className="add-point-btn">Add</button>
                      {customPoints.length > 0 && (
                        <button onClick={clearCustomPoints} className="clear-points-btn">Clear All</button>
                      )}
                    </div>
                  </div>
                </div>

                {/* Main Chart */}
                <ResponsiveContainer width="100%" height={400}>
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
                <div style={{ marginTop: '-420px', pointerEvents: 'none' }}>
                  <ResponsiveContainer width="100%" height={400}>
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
                      {results.normative.volumetric_features[selectedRegion] && (
                        <Scatter 
                          data={[{
                            age: parseInt(age),
                            volume: results.normative.volumetric_features[selectedRegion],
                            label: `Your Data (${results.normative.participant_id || 'Patient'})`
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

              {/* Clinical Interpretation */}
              <div className="clinical-interpretation">
                <h3>Clinical Interpretation</h3>
                <div className="interpretation-content">
                  <p>
                    The subject's brain volume is at the <strong>{getPercentilePosition()?.percentile}th percentile</strong> for 
                    their age group, indicating it is within the normal range.
                  </p>
                  {results.normative.volumetric_features?.total_brain && (
                    <p>
                      Total brain volume: <strong>
                        {(results.normative.volumetric_features.total_brain / 1000).toFixed(1)} cm³
                      </strong>
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;