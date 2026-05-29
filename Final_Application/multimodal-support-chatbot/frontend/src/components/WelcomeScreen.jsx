import React from 'react';
import { Sparkles } from 'lucide-react';

/**
 * WelcomeScreen — shown when there are no messages yet.
 * Clean centered layout with suggested prompts, matching Claude.ai's empty state.
 */
const SUGGESTIONS = [
  'Explain the Transformer architecture with diagrams',
  'How does multi-head attention work?',
  'Compare BERT and GPT architectures',
  'Show me the architecture of a diffusion model',
];

export default function WelcomeScreen({ onSuggestionClick }) {
  return (
    <div className="welcome-screen">
      <div className="welcome-content">
        <div className="welcome-icon">
          <Sparkles className="w-8 h-8" />
        </div>
        <h2 className="welcome-title">How can I help you today?</h2>
        <p className="welcome-subtitle">
          Ask me about technical documentation — I can retrieve text, diagrams, and schematics.
        </p>

        <div className="suggestion-grid">
          {SUGGESTIONS.map((s, i) => (
            <button
              key={i}
              className="suggestion-card"
              onClick={() => onSuggestionClick(s)}
            >
              <span className="suggestion-text">{s}</span>
              <span className="suggestion-arrow">→</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
