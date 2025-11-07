import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useKeycloak } from '@react-keycloak/web';
import PrivateRoute from './PrivateRoute';
// import SftpFiles from './SftpFiles';
import Worklist from './worklist';
import Dashboard from './Dashboard';
import './App.css';
import Navbar from './navbar.jsx';

function App() {
  const { keycloak, initialized } = useKeycloak();

  // Automatically redirect to Keycloak login if not authenticated and initialized
  useEffect(() => {
    if (initialized && !keycloak.authenticated) {
      keycloak.login();
    }
    // Persist username for API calls (e.g., start_processing, uploads)
    if (initialized && keycloak.authenticated) {
      const uname = keycloak?.tokenParsed?.preferred_username || keycloak?.tokenParsed?.email || '';
      if (uname) {
        try {
          localStorage.setItem('username', uname);
        } catch (_) {}
      }
    }
  }, [keycloak, initialized]);

  // Wait for Keycloak to initialize (including silent SSO check)
  if (!initialized) {
    return <div>Loading authentication...</div>;
  }

  // Render the app (authenticated users will see this)
  return (
    <Router>
      <Navbar keycloak={keycloak} />
      <Routes>
        <Route path="/" element={<div>Home (Public)</div>} />
        <Route
          path="/worklist"
          element={
            <PrivateRoute>
              <Worklist />
            </PrivateRoute>
          }
        />
        <Route
          path="/dashboard"
          element={
            <PrivateRoute>
              <Dashboard />
            </PrivateRoute>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;
