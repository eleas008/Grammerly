import React from 'react';

export default function LoadingBar({ segmentCount = 15 }) {
  return (
    <div className="loading-bar-container">
      <div className="segmented-loader">
        {Array.from({ length: segmentCount }).map((_, i) => (
          <div 
            key={i} 
            className="segment" 
            style={{ animationDelay: `${i * 0.1}s` }} 
          />
        ))}
      </div>
    </div>
  );
}