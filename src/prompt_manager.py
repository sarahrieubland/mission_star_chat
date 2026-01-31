from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import Optional
import os

# Import local prompts as fallback
import src.config_prompts as local_prompts


class PromptManager:
    """Manages prompts from LangSmith Hub with local fallback."""
    
    def __init__(self, handle: str = "", use_hub: bool = True, client: Optional[Client] = None):
        self.handle = handle  # Kept for backwards compatibility, but not used for private prompts
        self.use_hub = use_hub and bool(client)
        self.client = client
        self._cache = {}
        
        verbose = os.getenv("VERBOSE", "true").lower() == "true"
        if verbose:
            if self.use_hub:
                print(f"📝 PromptManager: Using LangSmith Hub")
            else:
                print(f"📝 PromptManager: Using local prompts (Hub disabled or no client)")
    
    def _extract_template_from_prompt(self, prompt) -> str:
        """
        Extract template string from various LangChain prompt objects.
        
        Handles:
        - PromptTemplate: has .template attribute
        - ChatPromptTemplate: has .messages list with prompt templates
        - Other structures: fallback to string conversion
        """
        # Direct PromptTemplate
        if hasattr(prompt, 'template') and isinstance(prompt.template, str):
            return prompt.template
        
        # ChatPromptTemplate
        if hasattr(prompt, 'messages') and prompt.messages:
            # Try to get the first human/user message template
            for msg in prompt.messages:
                # Message with prompt attribute
                if hasattr(msg, 'prompt'):
                    if hasattr(msg.prompt, 'template'):
                        return msg.prompt.template
                # Direct template attribute on message
                elif hasattr(msg, 'template'):
                    return msg.template
                # Content attribute
                elif hasattr(msg, 'content') and isinstance(msg.content, str):
                    return msg.content
            
            # If no template found in messages, try to reconstruct
            # This handles cases where messages is a list of tuples
            if prompt.messages and len(prompt.messages) > 0:
                first_msg = prompt.messages[0]
                if isinstance(first_msg, tuple) and len(first_msg) >= 2:
                    return first_msg[1]  # (role, content) tuple
        
        # Fallback: convert to string
        return str(prompt)
    
    def get_prompt(self, prompt_name: str, fallback: str = "") -> str:
        """
        Get a prompt from LangSmith Hub or fall back to local.
        
        Args:
            prompt_name: Name of the prompt (e.g., "star-situation-prompt")
                        For private prompts, just use the name.
                        For public prompts, use "handle/prompt-name" format.
            fallback: Local fallback prompt string
            
        Returns:
            The prompt template string
        """
        # Check cache first
        if prompt_name in self._cache:
            return self._cache[prompt_name]
        
        prompt_template = fallback
        verbose = os.getenv("VERBOSE", "true").lower() == "true"
        
        if self.use_hub and self.client:
            try:
                if verbose:
                    print(f"   📥 Pulling prompt from Hub: {prompt_name}")
                
                # Pull from LangSmith Hub
                prompt = self.client.pull_prompt(prompt_name)
                
                # Extract the template string
                prompt_template = self._extract_template_from_prompt(prompt)
                
                if verbose:
                    print(f"   ✅ Loaded prompt from Hub: {prompt_name}")
                    
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ Failed to load prompt '{prompt_name}' from Hub: {e}")
                    print(f"   📝 Using local fallback")
                prompt_template = fallback
        else:
            if verbose and prompt_name not in self._cache:
                print(f"   📝 Using local prompt: {prompt_name}")
        
        # Cache the result
        self._cache[prompt_name] = prompt_template
        return prompt_template
    
    def push_prompt(
        self, 
        prompt_name: str, 
        prompt_template: str, 
        description: str = "",
        is_public: bool = False
    ) -> bool:
        """
        Push a prompt to LangSmith Hub.
        
        Args:
            prompt_name: Name for the prompt (e.g., "star-situation-prompt")
                        For private prompts, just use the name.
                        For public prompts, you'd use "handle/prompt-name" format.
            prompt_template: The prompt template string
            description: Optional description
            is_public: Whether to make the prompt public
            
        Returns:
            True if successful, False otherwise
        """
        if not self.use_hub or not self.client:
            verbose = os.getenv("VERBOSE", "true").lower() == "true"
            if verbose:
                print(f"   ⚠️ Cannot push prompt: Hub disabled or client not initialized")
            return False
        
        try:
            verbose = os.getenv("VERBOSE", "true").lower() == "true"
            
            # Create a PromptTemplate object
            prompt = PromptTemplate.from_template(prompt_template)
            
            # Push to hub - no handle prefix needed for private prompts
            self.client.push_prompt(
                prompt_name, 
                object=prompt,
                description=description,
                is_public=is_public
            )
            
            if verbose:
                print(f"   ✅ Pushed prompt to Hub: {prompt_name}")
            return True
            
        except Exception as e:
            verbose = os.getenv("VERBOSE", "true").lower() == "true"
            if verbose:
                print(f"   ❌ Failed to push prompt '{prompt_name}': {e}")
                print(f"   Error type: {type(e).__name__}")
            return False
    
    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache = {}
        verbose = os.getenv("VERBOSE", "true").lower() == "true"
        if verbose:
            print("   🗑️ Prompt cache cleared")
    
    # Convenience properties for each prompt
    @property
    def AGENT_SYSTEM_PROMPT(self) -> str:
        from . import config_prompts as local_prompts
        return self.get_prompt("star-agent-system", local_prompts.AGENT_SYSTEM_PROMPT)
    
    @property
    def SITUATION_PROMPT(self) -> str:
        from . import config_prompts as local_prompts
        return self.get_prompt("star-situation", local_prompts.SITUATION_PROMPT)
    
    @property
    def TASK_PROMPT(self) -> str:
        from . import config_prompts as local_prompts
        return self.get_prompt("star-task", local_prompts.TASK_PROMPT)
    
    @property
    def ACTION_PROMPT(self) -> str:
        from . import config_prompts as local_prompts
        return self.get_prompt("star-action", local_prompts.ACTION_PROMPT)
    
    @property
    def RESULT_PROMPT(self) -> str:
        from . import config_prompts as local_prompts
        return self.get_prompt("star-result", local_prompts.RESULT_PROMPT)
    
    @property
    def GENERATE_STAR_PROMPT(self) -> str:
        from . import config_prompts as local_prompts
        return self.get_prompt("star-generate", local_prompts.GENERATE_STAR_PROMPT)
    
    @property
    def EVALUATE_PROMPT(self) -> str:
        from . import config_prompts as local_prompts
        return self.get_prompt("star-evaluate", local_prompts.EVALUATE_PROMPT)
    
    @property
    def WELCOME_MESSAGE(self) -> str:
        from . import config_prompts as local_prompts
        return self.get_prompt("star-welcome", local_prompts.WELCOME_MESSAGE)
    
    @property
    def EXTRACTION_PROMPT(self) -> str:
        from . import config_prompts as local_prompts
        return self.get_prompt("star-extraction", local_prompts.EXTRACTION_PROMPT)


# --- Helper function to push all prompts to Hub ---
def push_all_prompts_to_hub(ls_client):
    """
    One-time function to push all local prompts to LangSmith Hub.
    Run this once to set up your prompts in the Hub.
    
    Usage:
        from prompt_manager import push_all_prompts_to_hub
        from langsmith import Client
        ls_client = Client()
        push_all_prompts_to_hub(ls_client)
    """
    if not ls_client:
        print("❌ LangSmith client not initialized. Check your LANGSMITH_API_KEY")
        return
    
    print("📤 Pushing all prompts to LangSmith Hub...")
    
    prompt_configs = [
        ("star-agent-system", local_prompts.AGENT_SYSTEM_PROMPT, "System prompt for STAR career coach agent"),
        ("star-situation", local_prompts.SITUATION_PROMPT, "Prompt to ask about the Situation component"),
        ("star-task", local_prompts.TASK_PROMPT, "Prompt to ask about the Task component"),
        ("star-action", local_prompts.ACTION_PROMPT, "Prompt to ask about the Action component"),
        ("star-result", local_prompts.RESULT_PROMPT, "Prompt to ask about the Result component"),
        ("star-generate", local_prompts.GENERATE_STAR_PROMPT, "Prompt to generate STAR text from components"),
        ("star-evaluate", local_prompts.EVALUATE_PROMPT, "Prompt to evaluate STAR text quality"),
        ("star-welcome", local_prompts.WELCOME_MESSAGE, "Welcome message for the STAR generator"),
        ("star-extraction", local_prompts.EXTRACTION_PROMPT, "Prompt to extract STAR components from initial job description"),
    ]
    
    pm = PromptManager(use_hub=True, client=ls_client)
    
    success_count = 0
    for name, template, description in prompt_configs:
        success = pm.push_prompt(name, template, description)
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
        if success:
            success_count += 1
    
    print(f"\n📊 Successfully pushed {success_count}/{len(prompt_configs)} prompts")
    print("\n✅ Done! You can now view and edit your prompts in LangSmith:")
    print(f"   https://smith.langchain.com/prompts")


# Example usage and testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize client
    try:
        ls_client = Client()
        
        print(f"Testing LangSmith prompt management...")
        
        # Create manager
        pm = PromptManager(use_hub=True, client=ls_client)
        
        # Test push
        test_template = "This is a test prompt with {variable}"
        success = pm.push_prompt("test-prompt", test_template, "Test prompt for debugging")
        
        if success:
            print("\n✅ Push successful!")
            
            # Test pull
            print("\n📥 Testing pull...")
            pm.clear_cache()  # Clear cache to force fresh pull
            retrieved = pm.get_prompt("test-prompt", fallback="FALLBACK")
            
            print(f"\n📋 Retrieved template: {retrieved}")
            print(f"✅ Match: {retrieved == test_template}")
        else:
            print("\n❌ Push failed - check your credentials")
            
    except Exception as e:
        print(f"❌ Error: {e}")