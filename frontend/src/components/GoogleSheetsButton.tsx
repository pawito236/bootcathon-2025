import React from 'react';
import { FileSpreadsheet } from 'lucide-react';

interface GoogleSheetsButtonProps {
  sheetsUrl: string;
}

const GoogleSheetsButton: React.FC<GoogleSheetsButtonProps> = ({ sheetsUrl }) => {
  const handleClick = () => {
    window.open(sheetsUrl, '_blank', 'noopener,noreferrer');
  };

  return (
    <button
      onClick={handleClick}
      className="fixed top-6 right-20 p-3 bg-green-600 text-white rounded-full shadow-lg hover:bg-green-700 hover:shadow-xl transform hover:scale-105 transition-all duration-200 z-10 group"
      title="Open Data Source"
    >
      <FileSpreadsheet className="w-5 h-5" />
      <div className="absolute -bottom-12 right-0 bg-gray-800 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap">
        Data Source
      </div>
    </button>
  );
};

export default GoogleSheetsButton;