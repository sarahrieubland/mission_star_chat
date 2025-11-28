import { Textarea } from "@/components/ui/textarea";
import React, { useState, useEffect, useRef } from 'react';

export default function EditableText() {
  // Access props from global scope (Chainlit injects them globally)
  const initialText = typeof props !== 'undefined' && props.initial ? props.initial : '';
  const [text, setText] = useState(initialText);
  const [saved, setSaved] = useState(false);
  const [hasEdited, setHasEdited] = useState(false);
  const observerRef = useRef(null);

  // Update from props when new text is generated (but preserve user edits if they're actively editing)
  useEffect(() => {
    if (props.initial && props.initial !== text) {
      // Only auto-update if user hasn't made recent edits
      if (!hasEdited) {
        setText(props.initial);
      }
    }
  }, [props.initial, text, hasEdited]);

  // Reset hasEdited flag when props change significantly (new generation)
  useEffect(() => {
    if (props.initial) {
      setHasEdited(false);
      setText(props.initial);
    }
  }, [props.initial]);

  const handleTextChange = (e) => {
    setText(e.target.value);
    setHasEdited(true);
  };

  useEffect(() => {
    // Inject CSS to constrain the main app
    const style = document.createElement('style');
    style.id = 'editable-text-styles';
    style.innerHTML = `
      #root {
        max-width: 50vw !important;
        width: 50vw !important;
      }
    `;
    
    // Only add if not already present
    if (!document.getElementById('editable-text-styles')) {
      document.head.appendChild(style);
    }
    
    // Observer to hide ONLY messages that contain the exact SAVE_STAR_TEXT: prefix
    // This is more targeted to avoid hiding legitimate messages
    observerRef.current = new MutationObserver(() => {
      // Find all message elements
      const elements = document.querySelectorAll('[class*="message"], .step, [class*="MessageContent"]');
      elements.forEach((el) => {
        const textContent = el.textContent || '';
        // Only hide if it starts with SAVE_STAR_TEXT: (the actual save command)
        // Be very specific to avoid hiding other messages
        if (textContent.trim().startsWith('SAVE_STAR_TEXT:')) {
          el.style.display = 'none';
        }
      });
    });
    
    observerRef.current.observe(document.body, { childList: true, subtree: true });
    
    return () => {
      const styleEl = document.getElementById('editable-text-styles');
      if (styleEl && styleEl.parentNode) {
        styleEl.parentNode.removeChild(styleEl);
      }
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, []);

  const handleSave = () => {
    console.log('[SAVE] Button clicked');
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
    
    // Find the chat input
    const chatInput = 
      document.querySelector('textarea[placeholder*="message"]') || 
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
      submitButton = container?.querySelector('button[type="submit"]') || null;
      console.log('[SAVE] Found button via container:', !!submitButton);
    }
    
    // Method 3: Look for any button with send icon near textarea
    if (!submitButton) {
      const allButtons = document.querySelectorAll('button');
      submitButton = Array.from(allButtons).find((btn) => {
        const svg = btn.querySelector('svg');
        const isNearInput = chatInput.parentElement?.contains(btn) || 
                           chatInput.parentElement?.parentElement?.contains(btn);
        return svg && isNearInput;
      }) || null;
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
    )?.set;
    
    if (nativeInputValueSetter) {
      nativeInputValueSetter.call(chatInput, `SAVE_STAR_TEXT:${text}`);
      console.log('[SAVE] Set new value with prefix');
      
      // Trigger input event
      const inputEvent = new Event('input', { bubbles: true });
      chatInput.dispatchEvent(inputEvent);
      console.log('[SAVE] Dispatched input event');
      
      // Capture submitButton in closure
      const buttonToClick = submitButton;
      
      // Wait a tiny bit for the button to become enabled
      setTimeout(() => {
        console.log('[SAVE] Attempting to click submit button');
        console.log('[SAVE] Button disabled status:', buttonToClick?.disabled);
        
        if (buttonToClick && !buttonToClick.disabled) {
          buttonToClick.click();
          console.log('[SAVE] Submit button clicked!');
        } else {
          console.error('[SAVE] Submit button is disabled, cannot click');
        }
        
        // Clear the input
        setTimeout(() => {
          if (nativeInputValueSetter) {
            nativeInputValueSetter.call(chatInput, originalValue);
            const clearEvent = new Event('input', { bubbles: true });
            chatInput.dispatchEvent(clearEvent);
            console.log('[SAVE] Input cleared');
          }
        }, 50);
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
            ✓ Sauvegardé
          </div>
        )}
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Texte STAR</h3>
        </div>
        <Textarea
          value={text}
          onChange={handleTextChange}
          className="flex-1 resize-none mb-4"
          placeholder="Le texte STAR apparaîtra ici au fur et à mesure..."
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
            Sauvegarder
          </button>
        </div>
      </div>
    </div>
  );
}