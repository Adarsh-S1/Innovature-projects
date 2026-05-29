import React from 'react';
import { cn } from '../lib/utils';
import MarkdownRenderer from './MarkdownRenderer';
import ImageGallery from './ImageGallery';

/**
 * ChatMessage — renders a single chat message (user or assistant).
 * User messages are right-aligned with a tinted background.
 * Assistant messages are left-aligned with full markdown + image support.
 * Matches Claude.ai's clean, avatar-free conversation layout.
 */
export default function ChatMessage({ message }) {
  const isUser = message.role === 'user';

  return (
    <div className={cn('chat-message', isUser ? 'chat-message--user' : 'chat-message--assistant')}>
      <div className="chat-message-inner">
        {/* Role label */}
        <div className="chat-role">
          {isUser ? 'You' : 'Nova AI'}
        </div>

        {/* Message content */}
        <div className={cn(
          'chat-bubble',
          isUser ? 'chat-bubble--user' : 'chat-bubble--assistant',
          message.isError && 'chat-bubble--error'
        )}>
          {isUser ? (
            <p className="user-text">{message.content}</p>
          ) : (
            <MarkdownRenderer content={message.content} />
          )}
        </div>

        {/* Retrieved images */}
        {!isUser && message.images && message.images.length > 0 && (
          <ImageGallery images={message.images} />
        )}

        {/* Sources & metadata strip */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="chat-sources">
            <span className="sources-label">Sources</span>
            <div className="sources-list">
              {message.sources.map((src, i) => (
                <span key={i} className="source-chip">
                  {src.source_file || src.doc_id}
                  {src.page_start != null && ` · p.${src.page_start}`}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Quality metadata */}
        {!isUser && message.metadata && message.metadata.quality_score != null && (
          <div className="chat-meta-strip">
            <span className={cn(
              'meta-badge',
              message.metadata.quality_score > 0.8 ? 'meta-badge--good' : 'meta-badge--fair'
            )}>
              Quality: {message.metadata.quality_score.toFixed(2)}
            </span>
            {message.metadata.confidence_level && (
              <span className="meta-badge meta-badge--neutral">
                {message.metadata.confidence_level}
              </span>
            )}
            {message.metadata.query_type && (
              <span className="meta-badge meta-badge--neutral">
                {message.metadata.query_type}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
