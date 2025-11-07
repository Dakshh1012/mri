import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Scatter, ScatterChart, Legend } from 'recharts';
import './Dashboard.css';

const Dashboard = () => {
  const location = useLocation();
  const { analysisResults, patientAge, patientGender } = location.state || {};
  
  const [results, setResults] = useState(null);
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  
  // Normative modeling interactive features
  const [selectedRegion, setSelectedRegion] = useState('total_brain');
  const [selectedPercentiles, setSelectedPercentiles] = useState([10, 25, 50, 75, 90]);
  const [customPoints, setCustomPoints] = useState([]);
  const [customAge, setCustomAge] = useState('');
  const [customVolume, setCustomVolume] = useState('');

  // Update results when navigation state changes
  useEffect(() => {
    if (analysisResults) {
      console.log('Received analysis results:', analysisResults);
      console.log('Normative data structure:', analysisResults.normative);
      console.log('Normative percentile_scores:', analysisResults.normative?.percentile_scores);
      console.log('Normative volumetric_features:', analysisResults.normative?.volumetric_features);
      setResults(analysisResults);
      if (patientAge) setAge(patientAge.toString());
      if (patientGender) setGender(patientGender);
    }
  }, [analysisResults, patientAge, patientGender]);

  // Update selectedRegion when results change to ensure it's valid
  useEffect(() => {
    if (!results) return;
    
    let availableRegions = [];
    
    // Try to get regions from percentile scores first
    if (results?.normative?.percentile_scores) {
      availableRegions = Object.keys(results.normative.percentile_scores);
      console.log("Available regions from percentile scores:", availableRegions);
    }
    
    // Fallback to volumetric features if no percentile scores
    if (availableRegions.length === 0 && results?.normative?.volumetric_features) {
      availableRegions = Object.keys(results.normative.volumetric_features);
      console.log("Available regions from volumetric features:", availableRegions);
    }
    
    // Update selectedRegion if current one is not available
    if (availableRegions.length > 0) {
      if (!availableRegions.includes(selectedRegion)) {
        const newRegion = availableRegions[0];
        console.log(`Updating selected region from ${selectedRegion} to ${newRegion}`);
        setSelectedRegion(newRegion);
      }
    } else {
      console.log("No regions available from API response");
    }
  }, [results, selectedRegion]);

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

  const generateNormativeData = (region) => {
    // Check if we have actual percentile curves from the API
    if (results?.normative?.percentile_curves?.[region]) {
      console.log("Using actual percentile curves for region:", region);
      const regionData = results.normative.percentile_curves[region];
      const ages = regionData.ages || [];
      const curves = regionData.percentile_curves || {};

      console.log(`Generating data from API for region: ${region}, ages: ${ages.length}, curves:`, Object.keys(curves));

      // Transform the API data into the format Recharts expects
      return ages.map((agePoint, index) => {
        const dataPoint = { age: agePoint };
        selectedPercentiles.forEach(percentile => {
          const percentileKey = percentile.toString();
          if (curves[percentileKey] && curves[percentileKey][index] !== undefined) {
            dataPoint[`p${percentile}`] = curves[percentileKey][index];
          }
        });
        return dataPoint;
      });
    }
    
    // Fallback: Generate synthetic curves if API doesn't provide them
    console.log("API percentile curves not available, using fallback for region:", region);
    console.log("Available normative data:", results?.normative);
    
    // Generate age range from 1 to 100 (dynamic as requested)
    const ages = Array.from({length: 100}, (_, i) => i + 1);
    
    // Get the user's actual percentile score for this region if available
    const userPercentile = results?.normative?.percentile_scores?.[region] || 50;
    const userVolume = results?.normative?.volumetric_features?.[region] || 0;
    
    console.log(`Region ${region}: User percentile = ${userPercentile}, User volume = ${userVolume}`);
    
    // Generate synthetic percentile curves based on typical brain volume patterns
    // This is a fallback when the backend doesn't provide actual curve data
    return ages.map(age => {
      const dataPoint = { age };
      
      // Generate realistic brain volume curves (in mm³)
      // These are approximations based on typical neurodevelopmental patterns
      const baseVolume = region.includes('total') ? 
        (age < 18 ? 800000 + (age * 15000) : 
         age < 60 ? 1300000 - ((age - 18) * 2000) : 
         1216000 - ((age - 60) * 3000)) :
        (age < 18 ? 80000 + (age * 1500) : 
         age < 60 ? 107000 - ((age - 18) * 200) : 
         98600 - ((age - 60) * 300));
      
      // Generate percentile curves around the base volume
      selectedPercentiles.forEach(percentile => {
        // Use z-score approximations for percentile curves
        let zScore;
        if (percentile === 50) zScore = 0;
        else if (percentile === 75) zScore = 0.67;
        else if (percentile === 90) zScore = 1.28;
        else if (percentile === 95) zScore = 1.64;
        else if (percentile === 25) zScore = -0.67;
        else if (percentile === 10) zScore = -1.28;
        else if (percentile === 5) zScore = -1.64;
        else zScore = (percentile - 50) / 15; // Rough approximation
        
        // Apply variation based on age (more variation in younger ages)
        const ageVariation = age < 20 ? 0.15 : 0.10;
        const stdDev = baseVolume * ageVariation;
        
        dataPoint[`p${percentile}`] = baseVolume + (zScore * stdDev);
      });
      
      return dataPoint;
    });
  };

  const getPercentilePosition = () => {
    // Use actual percentile scores from the API for the selected region
    if (!results?.normative?.percentile_scores?.[selectedRegion]) {
      console.log("No percentile scores found for region:", selectedRegion);
      console.log("Available percentile regions:", Object.keys(results?.normative?.percentile_scores || {}));
      return null;
    }
    
    const percentile = results.normative.percentile_scores[selectedRegion];
    
    // Assign colors based on percentile ranges
    let color;
    if (percentile >= 90) color = '#4CAF50';      // Green - high normal
    else if (percentile >= 75) color = '#8BC34A'; // Light green - upper normal
    else if (percentile >= 60) color = '#FFC107'; // Yellow - upper average
    else if (percentile >= 40) color = '#FF9800'; // Orange - lower average
    else if (percentile >= 25) color = '#FF5722'; // Red-orange - below average
    else color = '#F44336';                       // Red - low
    
    return { percentile: Math.round(percentile), color };
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
    // Prioritize regions that have actual percentile curves from API
    if (results?.normative?.percentile_curves && Object.keys(results.normative.percentile_curves).length > 0) {
      const curveRegions = Object.keys(results.normative.percentile_curves);
      console.log("Available regions from percentile curves:", curveRegions);
      return curveRegions;
    }
    
    // Fallback to regions that have percentile scores
    if (results?.normative?.percentile_scores && Object.keys(results.normative.percentile_scores).length > 0) {
      const scoreRegions = Object.keys(results.normative.percentile_scores);
      console.log("Available regions from percentile scores:", scoreRegions);
      return scoreRegions;
    }
    
    // Final fallback to volumetric features
    if (results?.normative?.volumetric_features) {
      const volumeRegions = Object.keys(results.normative.volumetric_features);
      console.log("Available regions from volumetric features:", volumeRegions);
      return volumeRegions;
    }
    
    console.log("No regions available from API response");
    return [];
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

  const openVisualizer = async () => {
  try {
    await fetch('http://localhost:7000/open-visualizer', { method: 'POST' });
    alert('Opening 3D Visualizer...');
  } catch (err) {
    console.error('Failed to open visualizer:', err);
    alert('Error: Could not open visualizer.');
  }
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
          <span> Analytics Dashboard</span>
        </div>
      </header>

      {!results ? (
        <div className="upload-section">
          <div className="upload-card">
            <h2>No Analysis Results Available</h2>
            <p>Please analyze a study from the Worklist to view results here.</p>
          </div>
        </div>
      ) : (
        <div className="results-dashboard">
          {/* Patient Information Header */}
          <div className="patient-info">
            <div className="info-item">
              <span className="label">Patient ID:</span>
              <span className="value">{results?.brainAge?.participant_id || results?.normative?.participant_id || 'N/A'}</span>
            </div>
            <div className="info-item">
              <span className="label">Age:</span>
              <span className="value">{age} years</span>
            </div>
            <div className="info-item">
              <span className="label">Sex:</span>
              <span className="value">{gender === 'M' ? 'Male' : gender === 'F' ? 'Female' : gender || 'N/A'}</span>
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
                <button className="viewer-btn" onClick={openVisualizer}>
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
                      {getAvailableRegions().length > 0 ? (
                        getAvailableRegions().map(region => (
                          <option key={region} value={region}>
                            {region.replace(/_/g, ' ').replace(/approximation/g, '').trim()}
                          </option>
                        ))
                      ) : (
                        <option value="">No regions available</option>
                      )}
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
                        domain={['dataMin', 'dataMax']}
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
                      {(() => {
                        // Get the user's actual volume for the selected region
                        const volumetricFeatures = results.normative?.volumetric_features || {};
                        let volume = volumetricFeatures[selectedRegion];
                        
                        // If exact region name not found, try common variations
                        if (!volume) {
                          const altRegion = selectedRegion.replace('_', ' ');
                          volume = volumetricFeatures[altRegion];
                        }
                        
                        // Get the user's percentile for this region
                        const percentileScore = results.normative?.percentile_scores?.[selectedRegion];
                        
                        console.log(`User data point - Region: ${selectedRegion}, Volume: ${volume}, Percentile: ${percentileScore}, Age: ${age}`);
                        
                        return volume && age ? (
                          <Scatter 
                            data={[{
                              age: parseInt(age),
                              volume: volume,
                              label: `Your Data (${results.normative.participant_id || 'Patient'}) - ${percentileScore ? percentileScore.toFixed(1) + 'th percentile' : 'N/A'}`
                            }]}
                            fill="#ff5722"
                            stroke="#fff"
                            strokeWidth={2}
                            r={8}
                          />
                        ) : null;
                      })()}
                      
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
                  {(() => {
                    const percentilePos = getPercentilePosition();
                    const zScore = results?.normative?.z_scores?.[selectedRegion];
                    const isOutlier = results?.normative?.outlier_regions?.includes(selectedRegion);
                    
                    return (
                      <>
                        {percentilePos ? (
                          <p>
                            The subject's <strong>{selectedRegion.replace(/_/g, ' ')}</strong> volume is at the{' '}
                            <strong style={{color: percentilePos.color}}>
                              {percentilePos.percentile}th percentile
                            </strong> for their age group.
                          </p>
                        ) : (
                          <p>Percentile data not available for the selected region.</p>
                        )}
                        
                        {zScore !== undefined && (
                          <p>
                            Z-score: <strong>{zScore.toFixed(2)}</strong>
                            {isOutlier && <span style={{color: '#ff5722'}}> (Outlier detected)</span>}
                          </p>
                        )}
                        
                        {results.normative?.volumetric_features?.[selectedRegion] && (
                          <p>
                            {selectedRegion.replace(/_/g, ' ')} volume:{' '}
                            <strong>
                              {(results.normative.volumetric_features[selectedRegion] / 1000).toFixed(1)} cm³
                            </strong>
                          </p>
                        )}
                        
                        {results.normative?.volumetric_features?.total_brain && selectedRegion !== 'total_brain' && (
                          <p>
                            Total brain volume:{' '}
                            <strong>
                              {(results.normative.volumetric_features.total_brain / 1000).toFixed(1)} cm³
                            </strong>
                          </p>
                        )}
                        
                        {results.normative?.outlier_regions?.length > 0 && (
                          <div style={{marginTop: '15px', padding: '10px', backgroundColor: '#fff3cd', borderRadius: '5px'}}>
                            <strong>Note:</strong> Outlier regions detected: {results.normative.outlier_regions.join(', ')}
                          </div>
                        )}
                      </>
                    );
                  })()}
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