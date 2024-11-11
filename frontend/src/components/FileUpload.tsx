import React, { useCallback } from 'react';
import { Upload } from 'lucide-react';

interface FileUploadProps {
  file: File | null;
  onFileSelect: (file: File) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ file, onFileSelect }) => {
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const files = e.dataTransfer.files;
    if (files?.[0]) {
      const droppedFile = files[0];
      if (droppedFile.type.startsWith('video/')) {
        onFileSelect(droppedFile);
      }
    }
  }, [onFileSelect]);

  const handleClick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'video/*';
    input.onchange = (e) => {
      const files = (e.target as HTMLInputElement).files;
      if (files?.[0]) {
        onFileSelect(files[0]);
      }
    };
    input.click();
  };

  return (
    <div
      onClick={handleClick}
      onDragEnter={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      className="input-file group"
    >
      <Upload className="w-8 h-8 mb-2 group-hover:scale-110 transition-transform duration-200" />
      <span className="text-sm mt-2 text-center">
        {file ? file.name : 'Drag and drop or click to select a video file'}
      </span>
    </div>
  );
};

export default FileUpload;