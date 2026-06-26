import React from 'react';
import { Link } from 'react-router-dom';
import { FiUpload, FiList } from 'react-icons/fi';
// pipeline illustration removed

interface HeroProps {
  onUploadClick: () => void;
}

export const Hero: React.FC<HeroProps> = ({ onUploadClick }) => {
  return (
    <section className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center p-12 md:p-16 bg-gradient-to-br from-indigo-900 to-slate-900 rounded-2xl border border-var(--color-border) relative overflow-hidden">
      {/* Left side – text and actions */}
      <div className="flex flex-col items-start gap-6 animate-fadeIn">
        <span className="text-xs font-bold uppercase tracking-wider text-indigo-300 bg-indigo-900/30 px-3 py-1 rounded">
          Clinical Screening Platform
        </span>
        <h1 className="text-4xl md:text-5xl font-extrabold text-white leading-tight">
          Early AI‑Assisted Autism Screening
        </h1>
        <p className="text-lg text-gray-300 max-w-md">
          Empowering clinicians with scale‑invariant keypoint extraction and recurrent neural networks (Bi‑LSTM) to perform objective motor behavior screening in minutes.
        </p>
        <div className="flex gap-4 mt-2">
          <button onClick={onUploadClick} className="btn-indigo flex items-center gap-2">
            <FiUpload size={18} /> Upload Video
          </button>
          <Link to="/history" className="btn-outline flex items-center gap-2">
            <FiList size={18} /> View History
          </Link>
        </div>
      </div>

      {/* Glowing background accent (preserved) */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-[-60px] left-1/2 w-64 h-32 bg-indigo-500/30 blur-3xl rounded-full -translate-x-1/2"></div>
      </div>
    </section>
  );
};

export default Hero;
