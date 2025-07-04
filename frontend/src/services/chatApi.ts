import { ApiRequest, ApiResponse } from '../types/chat';

const API_ENDPOINT = '/api/chat';

export const sendChatMessage = async (question: string): Promise<ApiResponse> => {
  try {
    const response = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question } as ApiRequest),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data: ApiResponse = await response.json();
    return data;
  } catch (error) {
    console.error('Chat API error:', error);
    throw new Error('Failed to get response from AI agent. Please try again.');
  }
};