import React from 'react';
import { Bot } from 'lucide-react';

const TypingIndicator: React.FC = () => {
  return (
    <div className="flex w-full mb-6 justify-start">
      <div className="flex max-w-[85%] flex-row gap-4">
        <div className="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center exxon-gradient shadow-md">
          <Bot className="w-5 h-5 text-white" />
        </div>
        
        <div className="exxon-card text-exxon-text rounded-2xl rounded-bl-md px-5 py-4 exxon-shadow border border-gray-100">
          <div className="flex items-center gap-3">
            <div className="flex gap-1">
              <div className="w-2 h-2 bg-exxon-light-blue rounded-full animate-pulse"></div>
              <div className="w-2 h-2 bg-exxon-light-blue rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-2 h-2 bg-exxon-light-blue rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
            </div>
            <span className="text-sm text-exxon-text/70 font-medium">AI is analyzing data...</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;