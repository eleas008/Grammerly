import React from 'react';
import { renderDiffText } from '../utils/diffRenderer';

export default function OutputDisplay({ outputText, inputText, selectedMode }) {
  if (!outputText) return null;

  const content = selectedMode === 'GRAMMER CORRECTION' 
    ? renderDiffText(inputText, outputText) 
    : outputText;

  return (
    <div className="output-wrapper">
      <div className="text-box output-box formatted-output">
        {content}
      </div>
    </div>
  );
}