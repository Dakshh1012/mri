import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navigation = () => {
  const location = useLocation();

  return (
    <nav className="navigation">
      <div className="nav-links">
        <Link 
          to="/brain-age" 
          className={`nav-link ${location.pathname === '/' || location.pathname === '/brain-age' ? 'active' : ''}`}
        >
          🧠 Brain Age Prediction
        </Link>
        <Link 
          to="/normative" 
          className={`nav-link ${location.pathname === '/normative' ? 'active' : ''}`}
        >
          📊 Normative Modeling
        </Link>
      </div>
    </nav>
  );
};

export default Navigation;