import React from 'react';
import { Send } from 'lucide-react';

export default function InputSection({ inputText, setInputText, onSend, isLoading }) {
  return (
    <div className="input-wrapper">
      <textarea
        className="text-box input-box"
        placeholder="Type or paste your text here..."
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
      />
      <button 
        className="send-btn" 
        onClick={onSend} 
        title="Send" 
        disabled={isLoading || !inputText.trim()}
      >
        <Send size={28} fill="#2563eb" color="#2563eb" />
      </button>
    </div>
  );
}