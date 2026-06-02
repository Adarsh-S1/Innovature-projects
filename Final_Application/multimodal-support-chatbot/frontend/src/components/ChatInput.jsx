import React, { useRef, useEffect } from 'react';
import { Send, Paperclip, Loader2 } from 'lucide-react';

/**
 * ChatInput — the message input area at the bottom.
 * Auto-growing textarea with send button.
 * Matches Claude.ai's centered, minimal input bar.
 */
export default function ChatInput({ value, onChange, onSubmit, isLoading, onFileUpload, isUploading }) {
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const disabled = isLoading || isUploading;

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }, [value]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSubmit(e);
    }
  };

  return (
    <div className="chat-input-container">
      <form onSubmit={onSubmit} className="chat-input-form">
        <div className="chat-input-box">
          <input
            type="file"
            accept="application/pdf"
            ref={fileInputRef}
            style={{ display: 'none' }}
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                onFileUpload(e.target.files[0]);
              }
              e.target.value = null;
            }}
          />
          <button
            type="button"
            className="chat-input-icon-btn"
            title="Upload PDF Document"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
          >
            <Paperclip className="w-5 h-5" />
          </button>

          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message Nova AI..."
            className="chat-textarea"
            rows={1}
            disabled={disabled}
          />

          <button
            type="submit"
            disabled={!value.trim() || disabled}
            className="chat-send-btn"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>

        <p className="chat-input-footer">
          Nova AI may make mistakes. Verify important information from source documents.
        </p>
      </form>
    </div>
  );
}
