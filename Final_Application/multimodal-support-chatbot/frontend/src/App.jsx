import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Send, Image as ImageIcon, Bot, User, Loader2 } from 'lucide-react';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

// Utility for Tailwind classes
export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(() => Math.random().toString(36).substring(7));
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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
        metadata: data.metadata || {}
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error while processing your request.',
          isError: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-slate-950 text-slate-50 font-sans selection:bg-indigo-500/30">
      {/* Sidebar / Info Panel */}
      <div className="w-80 border-r border-slate-800 bg-slate-900/50 p-6 hidden md:flex flex-col">
        <div className="flex items-center gap-3 mb-8">
          <div className="bg-indigo-500 p-2 rounded-xl shadow-lg shadow-indigo-500/20">
            <Bot className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
              Nova AI
            </h1>
            <p className="text-xs text-slate-400">Multimodal Technical Support</p>
          </div>
        </div>

        <div className="space-y-4 flex-1">
          <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
            <h3 className="text-sm font-semibold text-slate-300 mb-2">Capabilities</h3>
            <ul className="text-xs text-slate-400 space-y-2">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" /> Technical manuals
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" /> Vector search
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" /> Cross-modal image retrieval
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full bg-gradient-to-b from-slate-950 to-slate-900 relative">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6 scroll-smooth">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 space-y-4">
              <div className="w-16 h-16 rounded-2xl bg-slate-800/50 border border-slate-700/50 flex items-center justify-center mb-4">
                <Bot className="w-8 h-8 text-indigo-400" />
              </div>
              <h2 className="text-2xl font-medium text-slate-300">How can I help you today?</h2>
              <p className="max-w-md text-center text-sm">
                Ask me technical questions. I can search through manuals and retrieve relevant diagrams.
              </p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={idx}
                className={cn(
                  "flex gap-4 max-w-4xl mx-auto w-full",
                  msg.role === 'user' ? "flex-row-reverse" : "flex-row"
                )}
              >
                {/* Avatar */}
                <div className={cn(
                  "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1 border",
                  msg.role === 'user' 
                    ? "bg-indigo-600/20 border-indigo-500/30 text-indigo-400" 
                    : "bg-slate-800 border-slate-700 text-slate-300"
                )}>
                  {msg.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                </div>

                {/* Message Bubble */}
                <div className={cn(
                  "flex flex-col gap-3 max-w-[80%]",
                  msg.role === 'user' ? "items-end" : "items-start"
                )}>
                  <div className={cn(
                    "px-5 py-3.5 rounded-2xl shadow-sm text-[15px] leading-relaxed relative group",
                    msg.role === 'user'
                      ? "bg-indigo-600 text-white rounded-tr-sm"
                      : msg.isError
                        ? "bg-red-500/10 border border-red-500/20 text-red-400 rounded-tl-sm"
                        : "bg-slate-800/80 border border-slate-700/50 text-slate-200 rounded-tl-sm"
                  )}>
                    {msg.role === 'user' ? (
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    ) : (
                      <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-slate-900 prose-pre:border prose-pre:border-slate-800 max-w-none">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    )}
                  </div>

                  {/* Render Images if any */}
                  {msg.images && msg.images.length > 0 && (
                    <div className="flex flex-wrap gap-4 mt-2">
                      {msg.images.map((img, imgIdx) => (
                        <div key={imgIdx} className="relative group rounded-xl overflow-hidden border border-slate-700 bg-slate-800 p-2 max-w-sm">
                          <img 
                            src={img.url} 
                            alt={img.caption || 'Retrieved image'} 
                            className="rounded-lg w-full h-auto object-cover max-h-60"
                          />
                          <div className="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-3">
                            <p className="text-xs text-slate-200 line-clamp-3">{img.caption}</p>
                            <span className="text-[10px] text-indigo-300 mt-1 uppercase font-semibold">
                              Score: {img.relevance_score?.toFixed(2)}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Metadata Tag */}
                  {msg.metadata && Object.keys(msg.metadata).length > 0 && (
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider bg-slate-800/50 px-2 py-1 rounded-md border border-slate-700/50">
                        {msg.metadata.query_type}
                      </span>
                      {msg.metadata.quality_score && (
                        <span className={cn(
                          "text-[11px] font-medium uppercase tracking-wider px-2 py-1 rounded-md border",
                          msg.metadata.quality_score > 0.8 
                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" 
                            : "bg-amber-500/10 text-amber-400 border-amber-500/20"
                        )}>
                          Quality: {msg.metadata.quality_score.toFixed(2)}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          
          {isLoading && (
            <div className="flex gap-4 max-w-4xl mx-auto w-full">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-slate-800 border border-slate-700 text-slate-300 flex items-center justify-center mt-1">
                <Bot className="w-4 h-4" />
              </div>
              <div className="bg-slate-800/50 border border-slate-700/50 rounded-2xl rounded-tl-sm px-5 py-4 flex items-center gap-3 shadow-sm">
                <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
                <span className="text-sm text-slate-400 animate-pulse">Synthesizing answer...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 sm:p-6 bg-slate-950/80 backdrop-blur-md border-t border-slate-800/80">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500/20 to-cyan-500/20 rounded-2xl blur opacity-0 group-focus-within:opacity-100 transition duration-500"></div>
            <div className="relative flex items-end gap-2 bg-slate-900 border border-slate-700 shadow-xl rounded-2xl p-2 transition-all focus-within:border-indigo-500/50 focus-within:bg-slate-900/80">
              <button 
                type="button"
                className="p-3 text-slate-400 hover:text-indigo-400 hover:bg-indigo-500/10 rounded-xl transition-colors"
                title="Upload image (Demo)"
              >
                <ImageIcon className="w-5 h-5" />
              </button>
              
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                placeholder="Ask about the specifications, diagrams, or usage..."
                className="flex-1 max-h-32 min-h-[44px] bg-transparent resize-none outline-none py-3 text-slate-200 placeholder:text-slate-500 text-[15px]"
                rows={1}
              />
              
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                className="p-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-indigo-600/20"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
            <p className="text-center text-[11px] text-slate-500 mt-3 font-medium tracking-wide">
              NOVA AI MULTIMODAL SUPPORT BOT V1.0
            </p>
          </form>
        </div>
      </div>
    </div>
  );
}
