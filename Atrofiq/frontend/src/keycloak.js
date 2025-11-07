// src/keycloak.js
import Keycloak from 'keycloak-js';

export const keycloak = new Keycloak({
  url: 'http://localhost:8080', // Replace with your Keycloak server URL
  realm: 'Atrofiq', // Replace with your realm name
  clientId: 'Atrofiq', // Replace with your client ID
});

export const initOptions = {
  onLoad: 'login-required',
  pkceMethod: 'S256',
  checkLoginIframe: false,
  redirectUri: window.location.origin + '/worklist', // Redirect directly to worklist after login
};