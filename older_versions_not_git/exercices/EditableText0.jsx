import { Textarea } from "@/components/ui/textarea";
import React, { useState, useEffect } from 'react';

export default function EditableText() {
  // Access props from global scope (Chainlit injects them globally)
  const initialText = typeof props !== 'undefined' && props.initial ? props.initial : '';
  const [text, setText] = useState(initialText);
  const [saved, setSaved] = useState(false);
  const [hasEdited, setHasEdited] = useState(false);

  // Only update from props if user hasn't edited yet
  useEffect(() => {
    if (!hasEdited && props.initial) {
      setText(props.initial);
    }
  }, [props.initial, hasEdited]);

  const handleTextChange = (e) => {
    setText(e.target.value);
    setHasEdited(true);
  };

  useEffect(() => {
    // Inject CSS to constrain the main app and hide save messages
    const style = document.createElement('style');
    style.innerHTML = `
      #root {
        max-width: 50vw !important;
        width: 50vw !important;
      }
      
      /* Hide messages that start with SAVE_STAR_TEXT from chat history */
      .step:has([class*="content"]:first-child):has([class*="content"] > div > p:first-child) {
        &:has(p:first-child:is(:first-letter)) {
          /* Check if message starts with SAVE_STAR_TEXT */
        }
      }
      
      /* More aggressive - hide any message containing SAVE_STAR_TEXT */
      [class*="message"]:has(*:contains("SAVE_STAR_TEXT")),
      .step:has(*:contains("SAVE_STAR_TEXT")) {
        display: none !important;
      }
    `;
    document.head.appendChild(style);
    
    // Also actively remove SAVE_STAR_TEXT messages from DOM
    const observer = new MutationObserver(() => {
      document.querySelectorAll('[class*="message"], .step').forEach(el => {
        if (el.textContent.includes('SAVE_STAR_TEXT:')) {
          el.style.display = 'none';
        }
      });
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
    
    return () => {
      document.head.removeChild(style);
      observer.disconnect();
    };
  }, []);

  const handleSave = () => {
    console.log('[SAVE] Button clicked');
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
    
    // Find the chat input
    const chatInput = document.querySelector('textarea[placeholder*="message"]') || 
                      document.querySelector('textarea[placeholder*="Message"]') ||
                      document.querySelector('form textarea');
    
    console.log('[SAVE] Chat input found:', !!chatInput);
    
    if (!chatInput) {
      console.error('Could not find chat input');
      return;
    }
    
    // Try multiple ways to find the submit button
    let submitButton = null;
    
    // Method 1: Look for form
    const form = chatInput.closest('form');
    if (form) {
      submitButton = form.querySelector('button[type="submit"]');
      console.log('[SAVE] Found button via form');
    }
    
    // Method 2: Look in parent container
    if (!submitButton) {
      const container = chatInput.closest('div[class*="input"]') || chatInput.parentElement;
      submitButton = container?.querySelector('button[type="submit"]');
      console.log('[SAVE] Found button via container:', !!submitButton);
    }
    
    // Method 3: Look for any button with send icon near textarea
    if (!submitButton) {
      const allButtons = document.querySelectorAll('button');
      submitButton = Array.from(allButtons).find(btn => {
        const svg = btn.querySelector('svg');
        const isNearInput = chatInput.parentElement?.contains(btn) || 
                           chatInput.parentElement?.parentElement?.contains(btn);
        return svg && isNearInput;
      });
      console.log('[SAVE] Found button via SVG search:', !!submitButton);
    }
    
    // Method 4: Last resort - find ANY submit button
    if (!submitButton) {
      submitButton = document.querySelector('button[type="submit"]');
      console.log('[SAVE] Found button via document query:', !!submitButton);
    }
    
    console.log('[SAVE] Final submit button found:', !!submitButton);
    console.log('[SAVE] Submit button disabled:', submitButton?.disabled);
    
    if (!submitButton) {
      console.error('Could not find submit button after all methods');
      return;
    }
    
    // Store original value
    const originalValue = chatInput.value;
    console.log('[SAVE] Original value:', originalValue);
    
    // Set the value with our save prefix
    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype,
      'value'
    ).set;
    nativeInputValueSetter.call(chatInput, `SAVE_STAR_TEXT:${text}`);
    console.log('[SAVE] Set new value with prefix');
    
    // Trigger input event
    const inputEvent = new Event('input', { bubbles: true });
    chatInput.dispatchEvent(inputEvent);
    console.log('[SAVE] Dispatched input event');
    
    // Wait a tiny bit for the button to become enabled
    setTimeout(() => {
      console.log('[SAVE] Attempting to click submit button');
      console.log('[SAVE] Button disabled status:', submitButton.disabled);
      
      if (!submitButton.disabled) {
        submitButton.click();
        console.log('[SAVE] Submit button clicked!');
      } else {
        console.error('[SAVE] Submit button is disabled, cannot click');
      }
      
      // Clear the input
      setTimeout(() => {
        nativeInputValueSetter.call(chatInput, originalValue);
        const clearEvent = new Event('input', { bubbles: true });
        chatInput.dispatchEvent(clearEvent);
        console.log('[SAVE] Input cleared');
      }, 50);
    }, 100);
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
      <div className="flex flex-col w-full border rounded-lg p-4 shadow-lg" style={{ backgroundColor: 'var(--background)', position: 'relative' }}>
        {saved && (
          <div style={{
            position: 'absolute',
            top: '16px',
            right: '16px',
            padding: '8px 16px',
            backgroundColor: '#10b981',
            color: 'white',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: '500',
            zIndex: 10,
            boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
          }}>
            ✓ Saved
          </div>
        )}
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Edit Response</h3>
        </div>
        <Textarea
          value={text}
          onChange={handleTextChange}
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