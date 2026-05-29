import React, { useRef, useEffect } from 'react';
import { Send, Image as ImageIcon, Loader2 } from 'lucide-react';

/**
 * ChatInput — the message input area at the bottom.
 * Auto-growing textarea with send button.
 * Matches Claude.ai's centered, minimal input bar.
 */
export default function ChatInput({ value, onChange, onSubmit, isLoading }) {
  const textareaRef = useRef(null);

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
          <button
            type="button"
            className="chat-input-icon-btn"
            title="Upload image (Demo)"
          >
            <ImageIcon className="w-5 h-5" />
          </button>

          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message Nova AI..."
            className="chat-textarea"
            rows={1}
            disabled={isLoading}
          />

          <button
            type="submit"
            disabled={!value.trim() || isLoading}
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
