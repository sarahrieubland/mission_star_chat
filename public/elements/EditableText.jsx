import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import React, { useState, useEffect } from 'react';

export default function EditableText() {
  const [text, setText] = useState(props.initial || '');
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
          {saved && <span className="text-green-600 text-sm">✓ Saved</span>}
        </div>
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="flex-1 resize-none mb-4"
        />
        <div className="flex justify-end gap-2">
          <Button onClick={handleSave}>
            Save Changes
          </Button>
        </div>
      </div>
    </div>
  );
}