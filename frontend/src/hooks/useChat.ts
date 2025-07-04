import { useState, useCallback } from 'react';
import { ChatMessage } from '../types/chat';
import { sendChatMessage } from '../services/chatApi';

export const useChat = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addMessage = useCallback((message: ChatMessage) => {
    setMessages(prev => [...prev, message]);
  }, []);

  const sendMessage = useCallback(async (content: string) => {
    setError(null);
    setIsLoading(true);

    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content,
      timestamp: new Date(),
    };
    addMessage(userMessage);

    try {
      const response = await sendChatMessage(content);
      
      // Add AI response
      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        content: response.answer,
        timestamp: new Date(),
        chart: response.chart,
      };
      addMessage(aiMessage);
    } catch (err) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        content: err instanceof Error ? err.message : 'Sorry, I couldn\'t connect. Please try again.',
        timestamp: new Date(),
      };
      addMessage(errorMessage);
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  }, [addMessage]);

  const sendMockupMessage = useCallback((mockupData: any) => {
    setError(null);
    setIsLoading(true);

    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: mockupData.question,
      timestamp: new Date(),
    };
    addMessage(userMessage);

    // Simulate loading delay
    setTimeout(() => {
      // Add AI response with chart
      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        content: mockupData.answer,
        timestamp: new Date(),
        chartUrl: mockupData.chart,
      };
      addMessage(aiMessage);
      setIsLoading(false);
    }, 1500);
  }, [addMessage]);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    sendMockupMessage,
  };
};