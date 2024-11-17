import React from 'react';
import { HelpCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';

interface FAQItem {
  question: string;
  answer: string;
}

const FAQ = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const faqs: FAQItem[] = [
    {
      question: "How does the action detection system work?",
      answer: "Our action detection system uses a state-of-the-art YOLO (You Only Look Once) model trained on custom action datasets. It processes video frames in real-time, identifying and classifying human actions with high accuracy. The system analyzes spatial and temporal features to understand complex movements and behaviors."
    },
    {
      question: "What types of actions can the system detect?",
      answer: "The system is primarily trained to detect common human actions such as sitting, standing, walking, etc. For proximity relations, it can analyze spatial relationships between people and objects in the scene. The detection capabilities can be extended through additional training on specific actions."
    },
    {
      question: "What video formats are supported?",
      answer: "The system supports common video formats including MP4, AVI, and MOV. For optimal performance, we recommend using MP4 format with H.264 encoding. There's no strict limit on video resolution, but processing time may increase with higher resolutions."
    },
    {
      question: "How accurate is the detection system?",
      answer: "The system achieves high accuracy rates, typically above 80% for well-defined actions in clear conditions. Factors affecting accuracy include video quality, lighting conditions, occlusions, and complexity of actions. The confidence scores are provided in the analysis report for transparency."
    },
    {
      question: "Can I process videos in batch?",
      answer: "Currently, the system processes one video at a time to ensure optimal resource allocation and accuracy. However, you can queue multiple videos sequentially. We're working on implementing batch processing capabilities in future updates."
    },
    {
      question: "How is the processed data handled?",
      answer: "All uploaded videos are temporarily stored only for the duration of processing. Once you download the processed video and analysis report, the data is automatically removed from our servers. We prioritize user privacy through this temporary data storage approach."
    }
  ];

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
        <div className="flex items-center space-x-3 mb-8">
          <HelpCircle className="w-8 h-8 text-primary-600 dark:text-primary-400" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Frequently Asked Questions
          </h1>
        </div>

        <div className="space-y-4">
          {faqs.map((faq, index) => (
            <div
              key={index}
              className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden"
            >
              <button
                className="w-full px-6 py-4 text-left bg-gray-50 dark:bg-gray-700 
                         hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors 
                         duration-200 flex justify-between items-center"
                onClick={() => setOpenIndex(openIndex === index ? null : index)}
              >
                <span className="font-medium text-gray-900 dark:text-white">
                  {faq.question}
                </span>
                {openIndex === index ? (
                  <ChevronUp className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                )}
              </button>
              
              {openIndex === index && (
                <div className="px-6 py-4 bg-white dark:bg-gray-800">
                  <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                    {faq.answer}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FAQ;