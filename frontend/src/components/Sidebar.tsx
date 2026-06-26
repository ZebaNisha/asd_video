import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { FiHome, FiList, FiFileText, FiSettings, FiMoon, FiSun, FiMenu, FiLogOut } from 'react-icons/fi';
import './Sidebar.css';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  toggleDark: () => void;
  darkMode: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ isOpen, onToggle, toggleDark, darkMode }) => {
  const navigate = useNavigate();
  const navItems = [
    { to: '/', label: 'Dashboard', icon: <FiHome /> },
    { to: '/history', label: 'History', icon: <FiList /> },
    { to: '/reports', label: 'Reports', icon: <FiFileText /> },
    { to: '/settings', label: 'Settings', icon: <FiSettings /> },
  ];

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    navigate('/login');
  };

  return (
    <aside className={`sidebar ${isOpen ? 'open' : 'collapsed'}`}>
      {/* Sidebar Header with Brand and Toggle */}
      <div className="sidebar-header">
        <span className="sidebar-brand">ASD Detector</span>
        <button
          onClick={onToggle}
          className="sidebar-toggle-btn"
          aria-label="Toggle sidebar"
          title={isOpen ? 'Collapse Sidebar' : 'Expand Sidebar'}
        >
          <FiMenu size={18} />
        </button>
      </div>

      {/* Navigation Menu Links */}
      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => (isActive ? 'active' : '')}
            title={!isOpen ? item.label : undefined}
          >
            {item.icon}
            <span className={isOpen ? 'action-label' : 'hidden'}>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Sidebar Footer with Actions */}
      <div className="sidebar-footer">
        <button
          onClick={toggleDark}
          className="sidebar-action-btn"
          title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          aria-label="Toggle dark mode"
        >
          {darkMode ? <FiSun size={18} /> : <FiMoon size={18} />}
          <span className={isOpen ? 'action-label' : 'hidden'}>
            {darkMode ? 'Light Theme' : 'Dark Theme'}
          </span>
        </button>
        <button
          onClick={handleLogout}
          className="sidebar-action-btn logout-btn"
          title="Sign Out"
          aria-label="Sign out"
        >
          <FiLogOut size={18} />
          <span className={isOpen ? 'action-label' : 'hidden'}>Sign Out</span>
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
