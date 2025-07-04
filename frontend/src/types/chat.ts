export interface ChatMessage {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  chart?: any;
  chartUrl?: string;
}

export interface ApiResponse {
  answer: string;
  chart?: any;
}

export interface ApiRequest {
  question: string;
}