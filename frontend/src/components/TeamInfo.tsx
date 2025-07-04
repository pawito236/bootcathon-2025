import React, { useState } from 'react';
import { Info, X, Users, Trophy, Zap } from 'lucide-react';

const TeamInfo: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 p-3 bg-exxon-red text-white rounded-full shadow-lg hover:bg-red-700 hover:shadow-xl transform hover:scale-105 transition-all duration-200 z-20 group"
        title="Team Information"
      >
        <Info className="w-5 h-5" />
        <div className="absolute -top-12 right-0 bg-gray-800 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap">
          Team Info
        </div>
      </button>

      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="exxon-card rounded-2xl p-8 max-w-md w-full exxon-shadow relative">
            <button
              onClick={() => setIsOpen(false)}
              className="absolute top-4 right-4 p-2 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="text-center mb-6">
              <div className="w-16 h-16 exxon-gradient rounded-full flex items-center justify-center mx-auto mb-4">
                <Trophy className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-exxon-dark-blue mb-2">ExxonMobil Bootcathon 2025</h2>
              <div className="bg-gradient-to-r from-exxon-red to-exxon-light-blue h-1 rounded-full w-20 mx-auto"></div>
            </div>

            <div className="space-y-4">
              <div className="flex items-center gap-3 p-3 bg-gradient-to-r from-exxon-red/10 to-exxon-light-blue/10 rounded-lg">
                <Users className="w-5 h-5 text-exxon-red" />
                <div>
                  <p className="font-semibold text-exxon-dark-blue">Team Name</p>
                  <p className="text-exxon-text text-sm">พี่วิชส่งมา</p>
                </div>
              </div>

              <div className="flex items-center gap-3 p-3 bg-gradient-to-r from-exxon-light-blue/10 to-exxon-red/10 rounded-lg">
                <Zap className="w-5 h-5 text-exxon-light-blue" />
                <div>
                  <p className="font-semibold text-exxon-dark-blue">Track</p>
                  <p className="text-exxon-text text-sm">GenAI Track</p>
                </div>
              </div>

              <div className="bg-exxon-gray/50 rounded-lg p-4 mt-6">
                <p className="text-exxon-text text-sm leading-relaxed">
                  This AI Analytics Chat Interface demonstrates advanced data visualization capabilities 
                  and intelligent insights generation for ExxonMobil's operational excellence initiatives.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default TeamInfo;