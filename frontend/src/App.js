import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import BrainAge from './pages/BrainAge';
import NormativeModeling from './pages/NormativeModeling';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/brain-age" element={<BrainAge />} />
          <Route path="/normative" element={<NormativeModeling />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;