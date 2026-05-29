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

  return (
    <div className="app-layout">
      <Sidebar onNewChat={handleNewChat} />

      <main className="chat-main">
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
        />
      </main>
    </div>
  );
}
