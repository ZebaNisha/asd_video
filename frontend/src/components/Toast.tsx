// src/components/Toast.tsx
import React, { createContext, useContext, useState, type ReactNode } from 'react';
import { FiCheckCircle, FiAlertTriangle, FiInfo } from 'react-icons/fi';

type Toast = {
  id: number;
  message: string;
  type: 'success' | 'error' | 'info';
};

type ToastContextType = {
  addToast: (msg: string, type?: Toast['type']) => void;
};

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export const ToastProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = (msg: string, type: Toast['type'] = 'info') => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message: msg, type }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4000);
  };

  const getToastIcon = (type: Toast['type']) => {
    switch (type) {
      case 'success':
        return <FiCheckCircle size={18} style={{ flexShrink: 0 }} />;
      case 'error':
        return <FiAlertTriangle size={18} style={{ flexShrink: 0 }} />;
      case 'info':
      default:
        return <FiInfo size={18} style={{ flexShrink: 0 }} />;
    }
  };

  return (
    <ToastContext.Provider value={{ addToast }}>
      {children}
      <div className="toast-container">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`toast ${t.type} animate-fadeIn`}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}
          >
            {getToastIcon(t.type)}
            <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>{t.message}</span>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
};

export const useToast = (): ToastContextType => {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
};
