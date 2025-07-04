import React, { useEffect, useRef } from 'react';
import ChatMessage from './components/ChatMessage';
import TypingIndicator from './components/TypingIndicator';
import ChatInput from './components/ChatInput';
import GoogleSheetsButton from './components/GoogleSheetsButton';
import MockupButton from './components/MockupButton';
import TeamInfo from './components/TeamInfo';
import { useChat } from './hooks/useChat';

// Configure your Google Sheets URL here
const GOOGLE_SHEETS_URL = 'https://docs.google.com/spreadsheets/d/1MHup5kU8vFTz4-L98tCC5yryTseMDbOLuNp6Bn7_yM4/edit?usp=sharing';

function App() {
  const { messages, isLoading, sendMessage, sendMockupMessage } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-exxon-gray via-white to-blue-50 flex flex-col font-inter">
      {/* Query Suggestions Button */}
      <MockupButton onMockupClick={sendMockupMessage} />
      
      {/* Google Sheets Button */}
      <GoogleSheetsButton sheetsUrl={GOOGLE_SHEETS_URL} />

      {/* Team Info */}
      <TeamInfo />
      
      {/* Header */}
      <div className="exxon-card border-b border-gray-200 py-6 px-6 exxon-shadow">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center gap-4 mb-2">
            <div className="w-12 h-12 bg-exxon-red rounded-lg flex items-center justify-center">
              <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
              </svg>
            </div>
            <div>
              <h1 className="text-3xl font-bold text-exxon-dark-blue">ExxonMobil AI Analytics</h1>
              <p className="text-exxon-text text-sm mt-1 font-medium">Intelligent Data Insights & Visualization Platform</p>
            </div>
          </div>
          <div className="bg-gradient-to-r from-exxon-red to-exxon-light-blue h-1 rounded-full w-full"></div>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className="flex-1 overflow-y-auto px-6 py-8">
          <div className="max-w-4xl mx-auto">
            {messages.length === 0 && (
              <div className="text-center py-16">
                <div className="exxon-card rounded-2xl p-10 exxon-shadow max-w-2xl mx-auto">
                  <div className="w-20 h-20 exxon-gradient rounded-full flex items-center justify-center mx-auto mb-6">
                    <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <h3 className="text-2xl font-bold text-exxon-dark-blue mb-4">Welcome to ExxonMobil AI Analytics</h3>
                  <p className="text-exxon-text text-base leading-relaxed mb-6 max-w-lg mx-auto">
                    Leverage advanced AI capabilities to analyze complex data patterns, generate insights, and create comprehensive visualizations for informed decision-making.
                  </p>
                  <div className="bg-gradient-to-r from-exxon-red/10 to-exxon-light-blue/10 border border-exxon-light-blue/20 rounded-xl p-4">
                    <p className="text-exxon-dark-blue text-sm font-semibold flex items-center justify-center gap-2">
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                      </svg>
                      Try the "Query Suggestions\" button to explore sample warehouse data visualizations
                    </p>
                  </div>
                </div>
              </div>
            )}
            
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            
            {isLoading && <TypingIndicator />}
            
            <div ref={messagesEndRef} />
          </div>
        </div>
      </div>

      {/* Input Area */}
      <ChatInput onSendMessage={sendMessage} disabled={isLoading} />
    </div>
  );
}

export default App;