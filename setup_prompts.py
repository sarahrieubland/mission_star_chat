#!/usr/bin/env python
"""
Setup Script for STAR Text Generator
=====================================

This script pushes all local prompts to LangSmith Hub.
Run this ONCE during initial setup.

After running this script, you can:
1. Edit prompts in the LangSmith UI at https://smith.langchain.com/prompts
2. Your app will automatically use the updated prompts from the Hub

Usage:
    python setup_prompts.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if LANGSMITH_API_KEY is set
if not os.getenv("LANGSMITH_API_KEY"):
    print("❌ Error: LANGSMITH_API_KEY not found in .env file")
    print("\nPlease:")
    print("1. Go to https://smith.langchain.com/settings")
    print("2. Create an API key")
    print("3. Add it to your .env file:")
    print("   LANGSMITH_API_KEY=lsv2_pt_...")
    sys.exit(1)

# Import after loading env vars
try:
    from langsmith import Client
    from src.prompt_manager import push_all_prompts_to_hub
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("\nPlease install required packages:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

def main():
    """Main setup function."""
    print("="*60)
    print("STAR TEXT GENERATOR - PROMPT SETUP")
    print("="*60)
    print("\nThis script will push all prompts to LangSmith Hub.")
    print("You only need to run this ONCE during initial setup.\n")
    
    # Ask for confirmation
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Setup cancelled.")
        sys.exit(0)
    
    print("\n" + "="*60)
    
    # Initialize LangSmith client
    try:
        ls_client = Client()
        print("✅ LangSmith client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize LangSmith client: {e}")
        print("\nPlease check your LANGSMITH_API_KEY in .env")
        sys.exit(1)
    
    # Push prompts to hub
    try:
        push_all_prompts_to_hub(ls_client)
    except Exception as e:
        print(f"\n❌ Error during prompt push: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. View your prompts: https://smith.langchain.com/prompts")
    print("2. Edit them in the LangSmith UI if needed")
    print("3. Run your app: chainlit run app.py")
    print("\nYour app will automatically use prompts from LangSmith Hub.")
    print("="*60)

if __name__ == "__main__":
    main()