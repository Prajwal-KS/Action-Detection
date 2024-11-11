import { useState, useRef } from 'react';
import axios from 'axios';
import { Play, Download, AlertCircle } from 'lucide-react';
import FileUpload from '../components/FileUpload';

const API_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

interface AnalysisReport {
  total_frames: number;
  processed_frames: number;
  detection_type: string;
  total_detections: number;
  average_detections_per_frame: number;
  video_duration: string;
  resolution: string;
  fps: number;
}

const ProcessPage = () => {
  const [file, setFile] = useState<File | null>(null);
  const [detectionType, setDetectionType] = useState<'action' | 'proximity'>('action');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processedVideo, setProcessedVideo] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisReport, setAnalysisReport] = useState<AnalysisReport | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setProcessedVideo(null);
    setError(null);
    setAnalysisReport(null);
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsProcessing(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('detection_type', detectionType);

    try {
      // Check server health
      const healthCheck = await api.get('/health');
      if (!healthCheck.data.models_status[detectionType]) {
        throw new Error(`${detectionType} model is not available`);
      }

      const response = await api.post('/upload_video/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        params: {
          detection_type: detectionType,
        },
        onUploadProgress: (progressEvent) => {
          const progress = progressEvent.total
            ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
            : 0;
          setUploadProgress(progress);
        },
      });

      if (response.data.filename) {
        const processedVideoUrl = `${API_URL}/outputs/${response.data.filename}`;
        setProcessedVideo(processedVideoUrl);
        setAnalysisReport(response.data.report);
      } else {
        throw new Error('No filename received from server');
      }
    } catch (err: any) {
      console.error('Upload error:', err);
      setError(
        err.response?.data?.detail || 
        err.message ||
        'An error occurred while processing the video.'
      );
    } finally {
      setIsProcessing(false);
      setUploadProgress(0);
    }
  };

  const handleDownload = async () => {
    if (!processedVideo) return;
    
    try {
      const response = await api.get(processedVideo.replace('/outputs/', '/download/'), {
        responseType: 'blob',
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `processed_${detectionType}_${file?.name || 'video.mp4'}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download error:', err);
      setError('Failed to download the video. Please try again.');
    }
  };

  const downloadReport = () => {
    if (!analysisReport) return;
    
    const reportText = Object.entries(analysisReport)
      .map(([key, value]) => `${key.replace(/_/g, ' ').toUpperCase()}: ${value}`)
      .join('\n');
    
    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis_report_${detectionType}_${file?.name || 'video'}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="card space-y-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-primary-600 to-primary-400 
                       bg-clip-text text-transparent mb-2">
            Video Processing
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Upload your video and select the detection type to begin processing
          </p>
        </div>

        <div className="space-y-6">
          {/* File Upload Section */}
          <div className="space-y-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Upload Video
            </label>
            <FileUpload file={file} onFileSelect={handleFileSelect} />
          </div>

          {/* Detection Type Selection */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Detection Type
            </label>
            <div className="flex space-x-4">
              <button
                onClick={() => setDetectionType('action')}
                className={`flex-1 py-3 px-4 rounded-lg transition-all duration-200 ${
                  detectionType === 'action'
                    ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/30'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                Action Detection
              </button>
              <button
                onClick={() => setDetectionType('proximity')}
                className={`flex-1 py-3 px-4 rounded-lg transition-all duration-200 ${
                  detectionType === 'proximity'
                    ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/30'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                Proximity Relation
              </button>
            </div>
          </div>

          {/* Process Button */}
          <button
            onClick={handleUpload}
            disabled={!file || isProcessing}
            className="btn-primary w-full py-3"
          >
            <Play className="w-4 h-4" />
            <span>{isProcessing ? 'Processing...' : 'Process Video'}</span>
          </button>

          {/* Progress Bar */}
          {isProcessing && uploadProgress > 0 && (
            <div className="relative pt-1">
              <div className="overflow-hidden h-2 text-xs flex rounded-full bg-gray-200 dark:bg-gray-700">
                <div
                  className="animate-pulse-slow shadow-lg bg-primary-600"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              <div className="text-center text-sm text-gray-600 dark:text-gray-400 mt-1">
                {uploadProgress}% uploaded
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="flex items-center space-x-2 text-red-600 dark:text-red-400 p-4 bg-red-50 dark:bg-red-900/30 rounded-lg">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>
      </div>

      {/* Processed Video Section */}
      {processedVideo && (
        <div className="card space-y-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Processed Video
          </h2>
          
          <div className="rounded-lg overflow-hidden shadow-xl">
            <video
              ref={videoRef}
              src={processedVideo}
              controls
              className="w-full"
            />
          </div>

          <div className="flex flex-wrap gap-4">
            <button
              onClick={handleDownload}
              className="btn-primary bg-green-600 hover:bg-green-700"
            >
              <Download className="w-4 h-4" />
              <span>Download Video</span>
            </button>
          </div>

          {/* Analysis Report */}
          {analysisReport && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                Analysis Report
              </h3>
              <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg border border-gray-200 dark:border-gray-600">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Detection Type:</span>
                      <span className="font-medium">{analysisReport.detection_type}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Total Frames:</span>
                      <span className="font-medium">{analysisReport.total_frames}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Video Duration:</span>
                      <span className="font-medium">{analysisReport.video_duration}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Resolution:</span>
                      <span className="font-medium">{analysisReport.resolution}</span>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Total Detections:</span>
                      <span className="font-medium">{analysisReport.total_detections}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Avg. Detections/Frame:</span>
                      <span className="font-medium">{analysisReport.average_detections_per_frame}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 dark:text-gray-400">FPS:</span>
                      <span className="font-medium">{analysisReport.fps}</span>
                    </div>
                  </div>
                </div>
              </div>
              <button
                onClick={downloadReport}
                className="btn-primary bg-blue-600 hover:bg-blue-700"
              >
                <Download className="w-4 h-4" />
                <span>Download Report</span>
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ProcessPage;