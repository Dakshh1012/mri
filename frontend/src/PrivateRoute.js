import React from 'react';
import { Navigate } from 'react-router-dom';
import { useKeycloak } from '@react-keycloak/web';

const PrivateRoute = ({ children }) => {
  const { keycloak } = useKeycloak();

  // If user is authenticated, render the child route
  // Else, redirect to login
  return keycloak.authenticated ? children : <Navigate to="/" />;
};

export default PrivateRoute;
