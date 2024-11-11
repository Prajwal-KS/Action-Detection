import React from 'react';
import { Brain, Cpu, GitBranch, Layers } from 'lucide-react';

const ModelDetails = () => {
  // Graph descriptions
  const graphDescriptions = {
    results: "Training metrics over epochs showing loss values, precision, recall, and mAP scores. The steady decrease in loss values and improvement in metrics indicates successful model training.",
    confusion_matrix: "Confusion matrix showing the model's prediction accuracy across different classes, helping identify where the model performs well and where it might need improvement.",
    f1_curve: "F1 score curve showing the harmonic mean of precision and recall, providing a balanced measure of the model's accuracy.",
    labels: "Distribution of training labels, showing the frequency of different classes in the dataset.",
    p_curve: "Precision curve showing the model's ability to avoid false positives across different confidence thresholds.",
    pr_curve: "Precision-Recall curve illustrating the tradeoff between precision and recall at various threshold settings.",
    r_curve: "Recall curve demonstrating the model's ability to find all relevant cases across different confidence thresholds."
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        {/* Header */}
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
          Model Architecture & Details
        </h1>

        {/* Model Information Grid */}
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

          {/* Technical Specifications */}
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

          {/* Training Results Section */}
          <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Training Results Overview
            </h3>
            <div className="space-y-4">
              <img 
                src="training-images/results.png"
                alt="Training Results"
                className="w-full rounded-lg shadow-lg"
              />
              <p className="text-gray-600 dark:text-gray-300 mt-4">
                {graphDescriptions.results}
              </p>
            </div>
          </div>

          {/* Grid of Smaller Graphs */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Confusion Matrix */}
            <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Confusion Matrix
              </h3>
              <img 
                src="training-images/confusion_matrix.png"
                alt="Confusion Matrix"
                className="w-full rounded-lg shadow-lg"
              />
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-4">
                {graphDescriptions.confusion_matrix}
              </p>
            </div>

            {/* F1 Curve */}
            <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                F1 Score Curve
              </h3>
              <img 
                src="training-images/F1_curve.png"
                alt="F1 Score Curve"
                className="w-full rounded-lg shadow-lg"
              />
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-4">
                {graphDescriptions.f1_curve}
              </p>
            </div>

            {/* Labels */}
            <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Label Distribution
              </h3>
              <img 
                src="training-images/labels.jpg"
                alt="Label Distribution"
                className="w-full rounded-lg shadow-lg"
              />
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-4">
                {graphDescriptions.labels}
              </p>
            </div>

            {/* P Curve */}
            <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Precision Curve
              </h3>
              <img 
                src="training-images/P_curve.png"
                alt="Precision Curve"
                className="w-full rounded-lg shadow-lg"
              />
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-4">
                {graphDescriptions.p_curve}
              </p>
            </div>

            {/* PR Curve */}
            <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Precision-Recall Curve
              </h3>
              <img 
                src="training-images/PR_curve.png"
                alt="Precision-Recall Curve"
                className="w-full rounded-lg shadow-lg"
              />
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-4">
                {graphDescriptions.pr_curve}
              </p>
            </div>

            {/* R Curve */}
            <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Recall Curve
              </h3>
              <img 
                src="training-images/R_curve.png"
                alt="Recall Curve"
                className="w-full rounded-lg shadow-lg"
              />
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-4">
                {graphDescriptions.r_curve}
              </p>
            </div>
          </div>

          {/* Training and Validation Batches */}
          <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
              Training Samples
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200">
                  Training Batch
                </h4>
                <img 
                  src="training-images/train_batch0.jpg"
                  alt="Training Batch Sample" 
                  className="w-full rounded-lg shadow-lg"
                />
              </div>
              <div>
                <h4 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200">
                  Validation Batch
                </h4>
                <img 
                  src="training-images/val_batch0_labels.jpg"
                  alt="Validation Batch Sample" 
                  className="w-full rounded-lg shadow-lg"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelDetails;