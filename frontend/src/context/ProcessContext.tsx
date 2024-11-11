import React, { createContext, useContext, useState } from 'react';

interface AnalysisReport {
  total_frames: number;
  processed_frames: number;
  detection_type: string;
  total_detections: number;
  average_detections_per_frame: number;
  video_duration: string;
  resolution: string;
  fps: number;
  processing_time: string;
  detection_confidence: number;
  detected_classes: { [key: string]: number };
  elapsedTime: string;
  performance_metrics: {
    cpu_usage: number;
    memory_usage: number;
    processing_speed: number;
  };
}

interface ProcessContextType {
  file: File | null;
  setFile: (file: File | null) => void;
  detectionType: 'action' | 'proximity';
  setDetectionType: (type: 'action' | 'proximity') => void;
  uploadProgress: number;
  setUploadProgress: (progress: number) => void;
  processedVideo: string | null;
  setProcessedVideo: (url: string | null) => void;
  isProcessing: boolean;
  setIsProcessing: (processing: boolean) => void;
  error: string | null;
  setError: (error: string | null) => void;
  analysisReport: AnalysisReport | null;
  setAnalysisReport: (report: AnalysisReport | null) => void;
}

const ProcessContext = createContext<ProcessContextType | undefined>(undefined);

export const ProcessProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [file, setFile] = useState<File | null>(null);
  const [detectionType, setDetectionType] = useState<'action' | 'proximity'>('action');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processedVideo, setProcessedVideo] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisReport, setAnalysisReport] = useState<AnalysisReport | null>(null);

  return (
    <ProcessContext.Provider
      value={{
        file,
        setFile,
        detectionType,
        setDetectionType,
        uploadProgress,
        setUploadProgress,
        processedVideo,
        setProcessedVideo,
        isProcessing,
        setIsProcessing,
        error,
        setError,
        analysisReport,
        setAnalysisReport,
      }}
    >
      {children}
    </ProcessContext.Provider>
  );
};

export const useProcess = () => {
  const context = useContext(ProcessContext);
  if (context === undefined) {
    throw new Error('useProcess must be used within a ProcessProvider');
  }
  return context;
};