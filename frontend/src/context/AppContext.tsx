import React, { createContext, useReducer, useEffect } from 'react';
import type { Dispatch, ReactNode } from 'react';

// Define state shape
export interface AppState {
  darkMode: boolean;
  sidebarOpen: boolean;
  activeJobId: string | null;
  activeJobStatus: string | null;
  pipelineStep: number; // 0 = none, increments as pipeline progresses
  predictionResult?: {
    id?: string;
    diagnosis?: string;
    confidence?: number;
    model?: string;
    processingTime?: string;
    asdProbability?: number;
    tdProbability?: number;
    rawClassification?: string;
    createdAt?: string;
  };
}

// Define actions
export type Action =
  | { type: 'TOGGLE_DARK_MODE' }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'SET_DARK_MODE'; payload: boolean }
  | { type: 'SET_SIDEBAR_OPEN'; payload: boolean }
  | { type: 'SET_ACTIVE_JOB'; payload: { id: string | null; status: string | null } }
  | { type: 'SET_PIPELINE_STEP'; payload: number }
  | { type: 'SET_PREDICTION_RESULT'; payload: any }
  | { type: 'RESET_ACTIVE_JOB' };

const initialState: AppState = {
  darkMode: true, // Dark mode first by default
  sidebarOpen: true,
  activeJobId: null,
  activeJobStatus: null,
  pipelineStep: 0,
};

function appReducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'TOGGLE_DARK_MODE': {
      const newDark = !state.darkMode;
      try { localStorage.setItem('darkMode', JSON.stringify(newDark)); } catch (e) {}
      return { ...state, darkMode: newDark };
    }
    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarOpen: !state.sidebarOpen };
    case 'SET_DARK_MODE':
      return { ...state, darkMode: action.payload };
    case 'SET_SIDEBAR_OPEN':
      return { ...state, sidebarOpen: action.payload };
    case 'SET_ACTIVE_JOB':
      return { ...state, activeJobId: action.payload.id, activeJobStatus: action.payload.status };
    case 'SET_PIPELINE_STEP':
      return { ...state, pipelineStep: action.payload };
    case 'SET_PREDICTION_RESULT':
      return { ...state, predictionResult: action.payload };
    case 'RESET_ACTIVE_JOB':
      return {
        ...state,
        activeJobId: null,
        activeJobStatus: null,
        pipelineStep: 0,
        predictionResult: undefined
      };
    default:
      return state;
  }
}

// Context type
interface AppContextProps {
  state: AppState;
  dispatch: Dispatch<Action>;
}

export const AppContext = createContext<AppContextProps>({} as AppContextProps);

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Load persisted dark mode or system preference on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem('darkMode');
      if (stored !== null) {
        dispatch({ type: 'SET_DARK_MODE', payload: JSON.parse(stored) });
      } else {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        dispatch({ type: 'SET_DARK_MODE', payload: prefersDark });
      }
    } catch (e) {
      // ignore errors
    }
  }, []);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
};
