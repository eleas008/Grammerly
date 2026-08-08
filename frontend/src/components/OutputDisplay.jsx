import React from 'react';
import { renderDiffText } from '../utils/diffRenderer';

export default function OutputDisplay({ outputText, inputText, selectedMode }) {
  if (!outputText) return null;

  const cleanOutput = outputText.replace(/-NONE-/gi, '').trim();

  const content = selectedMode === 'GRAMMER CORRECTION' 
    ? renderDiffText(inputText, cleanOutput) 
    : cleanOutput;

  return (
    <div className="output-wrapper">
      <div className="text-box output-box formatted-output">
        {content}
      </div>
    </div>
  );
}