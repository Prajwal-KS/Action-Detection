import React from 'react';
import { Brain, Cpu, GitBranch, Layers } from 'lucide-react';

const ModelDetails = () => {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
          Model Architecture & Details
        </h1>

        <div className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
              <div className="flex items-center space-x-3 mb-4">
                <Brain className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Model Type
                </h2>
              </div>
              <p className="text-gray-600 dark:text-gray-300">
                YOLO (You Only Look Once) - Real-time object detection system
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
              <div className="flex items-center space-x-3 mb-4">
                <Layers className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Architecture
                </h2>
              </div>
              <p className="text-gray-600 dark:text-gray-300">
                Deep CNN with multiple detection layers for efficient object detection
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
              <div className="flex items-center space-x-3 mb-4">
                <Cpu className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Performance
                </h2>
              </div>
              <p className="text-gray-600 dark:text-gray-300">
                Real-time processing capabilities with high accuracy in action detection
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
              <div className="flex items-center space-x-3 mb-4">
                <GitBranch className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Training
                </h2>
              </div>
              <p className="text-gray-600 dark:text-gray-300">
                Custom trained on action detection dataset with optimized parameters
              </p>
            </div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Technical Specifications
            </h2>
            <ul className="list-disc list-inside space-y-2 text-gray-600 dark:text-gray-300">
              <li>Framework: YOLO (Ultralytics implementation)</li>
              <li>Input Resolution: Dynamic support for various video resolutions</li>
              <li>Processing Speed: Real-time processing capabilities</li>
              <li>Output Format: Processed video with detected actions highlighted</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelDetails;