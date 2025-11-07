import React from 'react';
import { Link } from 'react-router-dom';
import './navbar.css';

export default function Navbar({ keycloak }) {
  const handleLogout = () => {
    keycloak.logout({ redirectUri: window.location.origin });
  };

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <Link to="/worklist" className="navbar-brand" aria-label="AlrofIQ Home">
          <img
            src="/prain.png"
            alt="AlrofIQ logo"
            className="navbar-logo"
            onError={(e) => { e.currentTarget.onerror = null; e.currentTarget.src = '/brain.png'; }}
          />
          <span className="navbar-title">AtrofiQ</span>
        </Link>
        {keycloak?.authenticated && (
          <button onClick={handleLogout} className="logout-btn">Logout</button>
        )}
      </div>
    </nav>
  );
}
