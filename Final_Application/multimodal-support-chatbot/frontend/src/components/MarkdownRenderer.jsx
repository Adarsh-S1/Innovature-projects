import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';

/**
 * MarkdownRenderer — renders markdown with syntax-highlighted code blocks,
 * GFM tables, and a copy-to-clipboard button on code fences.
 * Matches Claude.ai's clean, readable markdown output.
 */
export default function MarkdownRenderer({ content }) {
  const [copiedBlock, setCopiedBlock] = React.useState(null);

  const handleCopy = (code, idx) => {
    navigator.clipboard.writeText(code);
    setCopiedBlock(idx);
    setTimeout(() => setCopiedBlock(null), 2000);
  };

  let codeBlockIdx = 0;

  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            const codeString = String(children).replace(/\n$/, '');

            if (!inline && match) {
              const currentIdx = codeBlockIdx++;
              return (
                <div className="code-block-wrapper">
                  <div className="code-block-header">
                    <span className="code-lang">{match[1]}</span>
                    <button
                      onClick={() => handleCopy(codeString, currentIdx)}
                      className="code-copy-btn"
                      title="Copy code"
                    >
                      {copiedBlock === currentIdx ? (
                        <><Check className="w-3.5 h-3.5" /> Copied</>
                      ) : (
                        <><Copy className="w-3.5 h-3.5" /> Copy</>
                      )}
                    </button>
                  </div>
                  <SyntaxHighlighter
                    style={oneDark}
                    language={match[1]}
                    PreTag="div"
                    customStyle={{
                      margin: 0,
                      borderRadius: '0 0 8px 8px',
                      fontSize: '13px',
                      lineHeight: '1.5',
                    }}
                    {...props}
                  >
                    {codeString}
                  </SyntaxHighlighter>
                </div>
              );
            }

            if (!inline) {
              const currentIdx = codeBlockIdx++;
              return (
                <div className="code-block-wrapper">
                  <div className="code-block-header">
                    <span className="code-lang">text</span>
                    <button
                      onClick={() => handleCopy(codeString, currentIdx)}
                      className="code-copy-btn"
                      title="Copy code"
                    >
                      {copiedBlock === currentIdx ? (
                        <><Check className="w-3.5 h-3.5" /> Copied</>
                      ) : (
                        <><Copy className="w-3.5 h-3.5" /> Copy</>
                      )}
                    </button>
                  </div>
                  <pre className="code-block-plain"><code {...props}>{children}</code></pre>
                </div>
              );
            }

            return (
              <code className="inline-code" {...props}>
                {children}
              </code>
            );
          },
          table({ children }) {
            return (
              <div className="table-wrapper">
                <table>{children}</table>
              </div>
            );
          },
          a({ href, children }) {
            return (
              <a href={href} target="_blank" rel="noopener noreferrer" className="md-link">
                {children}
              </a>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
