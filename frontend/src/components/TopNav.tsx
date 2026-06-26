import React from 'react';
import { FiSun, FiMoon, FiSearch, FiBell } from 'react-icons/fi';

interface TopNavProps {
  darkMode: boolean;
  toggleDarkMode: () => void;
}

export const TopNav: React.FC<TopNavProps> = ({ darkMode, toggleDarkMode }) => {
  return (
    <header 
      className="glass-panel flex items-center justify-between px-6 py-3 rounded-xl shadow-md z-40"
      style={{
        margin: '16px 16px 0 16px',
        border: '1px solid var(--glass-border)',
        height: '70px',
      }}
    >
      {/* Brand Name */}
      <div className="text-xl md:text-2xl font-extrabold text-primary tracking-tight">
        ASD Detector
      </div>

      {/* Modern Search Bar */}
      <div className="relative w-64 hidden md:block">
        <span className="absolute inset-y-0 left-3 flex items-center text-muted">
          <FiSearch size={15} />
        </span>
        <input
          type="text"
          placeholder="Search patient, run ID, or reports..."
          className="input-field w-full pl-10 pr-4 py-2 text-sm rounded-lg"
          style={{
            backgroundColor: 'rgba(0,0,0,0.15)',
            border: '1px solid var(--color-border)',
            outline: 'none',
          }}
        />
      </div>

      {/* Action Utilities */}
      <div className="flex items-center gap-3">
        <button
          className="p-2 rounded-full hover:bg-gray-200 transition-colors"
          style={{
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
            color: 'var(--color-text-secondary)',
            border: 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '36px',
            height: '36px',
          }}
          title="Notifications"
        >
          <FiBell size={16} />
        </button>
        <button
          className="p-2 rounded-full hover:bg-gray-200 transition-colors"
          style={{
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
            color: 'var(--color-text-secondary)',
            border: 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '36px',
            height: '36px',
          }}
          onClick={toggleDarkMode}
          title={darkMode ? 'Switch to Light Theme' : 'Switch to Dark Theme'}
          aria-label="Toggle dark mode"
        >
          {darkMode ? <FiSun size={16} /> : <FiMoon size={16} />}
        </button>
      </div>
    </header>
  );
};

export default TopNav;
