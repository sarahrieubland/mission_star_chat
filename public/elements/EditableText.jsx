import { Textarea } from "@/components/ui/textarea";
import React, { useState, useEffect } from 'react';

export default function EditableText() {
  // Access props from global scope (Chainlit injects them globally)
  const initialText = typeof props !== 'undefined' && props.initial ? props.initial : '';
  const [text, setText] = useState(initialText);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    // Inject CSS to constrain the main app
    const style = document.createElement('style');
    style.innerHTML = `
      #root {
        max-width: 50vw !important;
        width: 50vw !important;
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
    
    // Find the chat input and submit button
    const chatInput = document.querySelector('textarea[placeholder*="message"]') || 
                      document.querySelector('textarea') ||
                      document.querySelector('input[type="text"]');
    
    if (chatInput) {
      // Set the value with our save prefix
      const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        'value'
      ).set;
      nativeInputValueSetter.call(chatInput, `SAVE_STAR_TEXT:${text}`);
      
      // Trigger change event
      const event = new Event('input', { bubbles: true });
      chatInput.dispatchEvent(event);
      
      // Find and click submit button
      setTimeout(() => {
        const submitButton = document.querySelector('button[type="submit"]') ||
                           document.querySelector('button[aria-label*="Send"]') ||
                           Array.from(document.querySelectorAll('button')).find(btn => 
                             btn.textContent.includes('Send') || btn.querySelector('svg')
                           );
        
        if (submitButton) {
          submitButton.click();
        }
      }, 100);
    }
  };

  return (
    <div 
      className="fixed flex gap-4 p-4" 
      style={{ 
        left: '50vw', 
        right: 0, 
        top: 0,
        bottom: 0,
        zIndex: 1000,
        backgroundColor: 'var(--background)'
      }}
    >
      <div className="flex flex-col w-full border rounded-lg p-4 shadow-lg" style={{ backgroundColor: 'var(--background)' }}>
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Edit Response</h3>
          {saved && (
            <span className="text-sm font-medium px-3 py-1 rounded-md" style={{ 
              backgroundColor: '#10b981', 
              color: 'white' 
            }}>
              ✓ Saved
            </span>
          )}
        </div>
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="flex-1 resize-none mb-4"
        />
        <div className="flex justify-end gap-2">
          <button 
            onClick={handleSave}
            style={{
              padding: '8px 16px',
              backgroundColor: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500'
            }}
          >
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
}