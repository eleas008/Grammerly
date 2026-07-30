import React, { useState } from 'react';
import './App.css';

import Header from './components/Header';
import InputSection from './components/InputSection';
import ModeSelector from './components/ModeSelector';
import LoadingBar from './components/LoadingBar';
import OutputDisplay from './components/OutputDisplay';
import { processText } from './services/api';

export default function App() {
  const [inputText, setInputText] = useState('');
  const [selectedMode, setSelectedMode] = useState('SUMMARY');
  const [isLoading, setIsLoading] = useState(false);
  const [outputText, setOutputText] = useState('');

  const handleSend = async () => {
    if (!inputText.trim()) return;

    setIsLoading(true);
    setOutputText('');

    try {
      const result = await processText(inputText, selectedMode);
      setOutputText(result);
    } catch (error) {
      console.error('Error processing text:', error);
      setOutputText('An error occurred while processing your request.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <Header />
      
      <InputSection 
        inputText={inputText}
        setInputText={setInputText}
        onSend={handleSend}
        isLoading={isLoading}
      />

      <ModeSelector 
        selectedMode={selectedMode}
        onSelectMode={setSelectedMode}
      />

      {isLoading && <LoadingBar />}

      {!isLoading && (
        <OutputDisplay 
          outputText={outputText}
          inputText={inputText}
          selectedMode={selectedMode}
        />
      )}
    </div>
  );
}