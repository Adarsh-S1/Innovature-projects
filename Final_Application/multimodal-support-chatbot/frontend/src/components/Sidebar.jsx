import React from 'react';
import { Bot, Sparkles } from 'lucide-react';

/**
 * Sidebar component — capabilities panel and branding.
 * Mirrors Claude.ai's minimal left sidebar with conversation context.
 */
export default function Sidebar({ onNewChat }) {
  return (
    <aside className="sidebar">
      {/* Brand Header */}
      <div className="sidebar-header">
        <div className="brand-icon">
          <Sparkles className="w-5 h-5" />
        </div>
        <div>
          <h1 className="brand-title">Nova AI</h1>
          <p className="brand-subtitle">Multimodal Support</p>
        </div>
      </div>

      {/* New Chat Button */}
      <button onClick={onNewChat} className="new-chat-btn">
        <span className="new-chat-icon">+</span>
        New chat
      </button>

      {/* Capabilities Card */}
      <div className="capabilities-card">
        <h3 className="capabilities-title">Capabilities</h3>
        <ul className="capabilities-list">
          <li>
            <span className="capability-dot capability-dot--green" />
            Search technical manuals
          </li>
          <li>
            <span className="capability-dot capability-dot--blue" />
            Hybrid vector + keyword search
          </li>
          <li>
            <span className="capability-dot capability-dot--amber" />
            Cross-modal image retrieval
          </li>
          <li>
            <span className="capability-dot capability-dot--purple" />
            AI quality-scored answers
          </li>
        </ul>
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <p className="sidebar-version">Nova AI v1.0</p>
      </div>
    </aside>
  );
}
