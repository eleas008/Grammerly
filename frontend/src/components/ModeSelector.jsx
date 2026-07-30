import React from 'react';

const MODES = ['GRAMMER CORRECTION', 'PARAPHRASE', 'SUMMARY'];

export default function ModeSelector({ selectedMode, onSelectMode }) {
  return (
    <div className="button-group">
      {MODES.map((mode) => (
        <button
          key={mode}
          className={`mode-btn ${selectedMode === mode ? 'active' : ''}`}
          onClick={() => onSelectMode(mode)}
        >
          {mode}
        </button>
      ))}
    </div>
  );
}