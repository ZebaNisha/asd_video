// src/components/Layout.tsx
import React, { useContext } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { ToastProvider } from './Toast';
import { TopNav } from './TopNav';
import { AppContext } from '../context/AppContext';

const Layout: React.FC = () => {
  const { state, dispatch } = useContext(AppContext);
  const { darkMode, sidebarOpen } = state;

  React.useEffect(() => {
    const theme = darkMode ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', theme);
  }, [darkMode]);

  const toggleDark = () => dispatch({ type: 'TOGGLE_DARK_MODE' });
  const toggleSidebar = () => dispatch({ type: 'TOGGLE_SIDEBAR' });

  return (
    <ToastProvider>
      <div 
        className="flex"
        style={{
          minHeight: '100vh',
          backgroundColor: 'var(--color-background)',
          color: 'var(--color-text-primary)',
          transition: 'background-color var(--transition-normal), color var(--transition-normal)',
          overflow: 'hidden',
        }}
      >
        <Sidebar isOpen={sidebarOpen} onToggle={toggleSidebar} toggleDark={toggleDark} darkMode={darkMode} />
        <div className="flex flex-col flex-1 overflow-hidden" style={{ height: '100vh' }}>
          <TopNav darkMode={darkMode} toggleDarkMode={toggleDark} />
          <main 
            className="flex-1"
            style={{
              padding: '24px 16px 16px 16px',
              overflowY: 'auto',
              maxHeight: 'calc(100vh - 102px)',
            }}
          >
            <div 
              style={{
                marginLeft: 'auto',
                marginRight: 'auto',
                maxWidth: '1300px',
                width: '100%',
              }}
            >
              <Outlet />
            </div>
          </main>
        </div>
      </div>
    </ToastProvider>
  );
};

export default Layout;
