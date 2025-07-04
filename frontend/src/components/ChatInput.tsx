import React, { useState, KeyboardEvent } from 'react';
import { Send } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, disabled }) => {
  const [message, setMessage] = useState('');

  const handleSend = () => {
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="p-6 exxon-card border-t border-gray-200 exxon-shadow">
      <div className="max-w-4xl mx-auto flex gap-4 items-end">
        <div className="flex-1 relative">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about data insights, trends, or request visualizations..."
            disabled={disabled}
            className="w-full px-6 py-4 pr-14 bg-white border-2 border-gray-200 rounded-full focus:outline-none focus:ring-2 focus:ring-exxon-light-blue focus:border-exxon-light-blue transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed text-exxon-text font-medium placeholder-gray-400"
          />
        </div>
        
        <button
          onClick={handleSend}
          disabled={disabled || !message.trim()}
          className="p-4 exxon-gradient text-white rounded-full hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-exxon-light-blue focus:ring-offset-2 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105"
        >
          <Send className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
};

export default ChatInput;