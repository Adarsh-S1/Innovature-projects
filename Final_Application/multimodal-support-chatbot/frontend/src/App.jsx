import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Loader2, Sparkles } from 'lucide-react';

import Sidebar from './components/Sidebar';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import WelcomeScreen from './components/WelcomeScreen';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(() =>
    Math.random().toString(36).substring(7)
  );
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // ── Submit handler ────────────────────────────────────────────────
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/api/v1/chat', {
        query: userMessage.content,
        session_id: sessionId,
      });

      const data = response.data;

      const botMessage = {
        role: 'assistant',
        content: data.answer,
        images: data.images || [],
        sources: data.sources || [],
        metadata: data.metadata || {},
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content:
            'Sorry, I encountered an error while processing your request. Please try again.',
          isError: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // ── New chat ──────────────────────────────────────────────────────
  const handleNewChat = () => {
    setMessages([]);
    setInput('');
    setSessionId(Math.random().toString(36).substring(7));
  };

  // ── Suggestion click ──────────────────────────────────────────────
  const handleSuggestionClick = (text) => {
    setInput(text);
  };

  // ── File Upload ───────────────────────────────────────────────────
  const handleFileUpload = async (file) => {
    if (file.type !== 'application/pdf') {
      alert("Only PDF files are supported");
      return;
    }
    
    setIsUploading(true);
    setUploadProgress(10);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await axios.post('http://localhost:8000/api/v1/ingest/pdf', formData);
      const jobId = res.data.job_id;
      
      // Poll for status
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await axios.get(`http://localhost:8000/api/v1/ingest/status/${jobId}`);
          const data = statusRes.data;
          
          if (data.status === 'success' || data.status === 'failed' || data.status === 'completed_with_errors') {
            clearInterval(pollInterval);
            setIsUploading(false);
            
            if (data.status === 'failed') {
               alert(data.errors?.[0] || 'Upload failed');
            } else {
               setMessages(prev => [...prev, {
                  role: 'assistant',
                  content: `Successfully processed **${file.name}**. Extracted ${data.total_chunks} text chunks and ${data.total_images} images. You can now ask questions about it!`,
               }]);
            }
          } else {
            setUploadProgress(prev => (prev < 90 ? prev + 10 : prev));
          }
        } catch (err) {
          clearInterval(pollInterval);
          setIsUploading(false);
          alert('Failed to check upload status');
        }
      }, 2000);
      
    } catch (error) {
      setIsUploading(false);
      alert(error.response?.data?.detail || 'Upload failed');
    }
  };

  return (
    <div className="app-layout">
      <Sidebar onNewChat={handleNewChat} />

      <main className="chat-main" style={{ position: 'relative' }}>
        {/* Upload Overlay */}
        {isUploading && (
          <div className="upload-overlay">
            <div className="upload-modal">
              <Loader2 className="w-8 h-8 animate-spin text-accent mb-4" />
              <h3 className="upload-title">Processing Document</h3>
              <p className="upload-subtitle">Extracting text and images...</p>
              <div className="upload-progress-bar">
                <div className="upload-progress-fill" style={{ width: `${uploadProgress}%` }} />
              </div>
            </div>
          </div>
        )}

        {/* Messages area */}
        <div className="chat-messages-area">
          {messages.length === 0 ? (
            <WelcomeScreen onSuggestionClick={handleSuggestionClick} />
          ) : (
            <>
              {messages.map((msg, idx) => (
                <ChatMessage key={idx} message={msg} />
              ))}

              {/* Loading indicator */}
              {isLoading && (
                <div className="chat-message chat-message--assistant">
                  <div className="chat-message-inner">
                    <div className="chat-role">Nova AI</div>
                    <div className="loading-indicator">
                      <div className="loading-dots">
                        <span /><span /><span />
                      </div>
                      <span className="loading-text">Thinking...</span>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <ChatInput
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          isLoading={isLoading}
          isUploading={isUploading}
          onFileUpload={handleFileUpload}
        />
      </main>
    </div>
  );
}
