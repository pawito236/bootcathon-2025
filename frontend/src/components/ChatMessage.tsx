import React from 'react';
import { ChatMessage as ChatMessageType } from '../types/chat';
import { User, Bot } from 'lucide-react';

interface ChatMessageProps {
  message: ChatMessageType;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.type === 'user';
  
  return (
    <div className={`flex w-full mb-6 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex max-w-[85%] ${isUser ? 'flex-row-reverse' : 'flex-row'} gap-4`}>
        <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
          isUser ? 'bg-exxon-red' : 'exxon-gradient'
        } shadow-md`}>
          {isUser ? (
            <User className="w-5 h-5 text-white" />
          ) : (
            <Bot className="w-5 h-5 text-white" />
          )}
        </div>
        
        <div className={`rounded-2xl px-5 py-4 ${
          isUser 
            ? 'bg-exxon-red text-white rounded-br-md shadow-lg' 
            : 'exxon-card text-exxon-text rounded-bl-md exxon-shadow border border-gray-100'
        }`}>
          <p className="text-sm leading-relaxed whitespace-pre-wrap font-medium">{message.content}</p>
          
          {/* Handle chart from API response (original format) */}
          {message.chart && (
            <div className="mt-4">
              <img 
                src={`https://quickchart.io/chart?c=${encodeURIComponent(JSON.stringify(message.chart))}`}
                alt="Chart visualization"
                className="rounded-xl max-w-full h-auto border border-gray-200 shadow-sm"
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                }}
              />
            </div>
          )}
          
          {/* Handle direct chart URL (mockup format) */}
          {message.chartUrl && (
            <div className="mt-4">
              <img 
                src={message.chartUrl}
                alt="Chart visualization"
                className="rounded-xl max-w-full h-auto border border-gray-200 shadow-sm"
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                }}
              />
            </div>
          )}
          
          <div className={`mt-3 text-xs font-medium ${isUser ? 'text-white/80' : 'text-exxon-text/60'}`}>
            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;