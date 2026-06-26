import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import { AppProvider } from './context/AppContext';
import Login from './pages/Login';
import ProtectedRoute from './components/ProtectedRoute';
import Overview from './pages/Overview';
import History from './pages/History';
import { Reports } from './pages/Reports';
import Settings from './pages/Settings';
import Instructions from './pages/Instructions';

const App: React.FC = () => (
  <AppProvider>
    <Router>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
            <Route index element={<Overview />} />
            <Route path="history" element={<History />} />
            <Route path="reports" element={<Reports />} />
            <Route path="instructions" element={<Instructions/>} />
            <Route path="settings" element={<Settings />} />
          </Route>
        </Routes>
    </Router>
  </AppProvider>
);

export default App;
