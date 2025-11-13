"""
Google Gemini wrapper using the modern Google Gen AI SDK (google-genai).

This wrapper uses the new unified Google Gen AI SDK which provides:
- Better system instruction support
- Modern API structure
- Unified interface across Google AI services
- Improved error handling and response parsing

Requires: google-genai>=1.0.0
"""
from google import genai
from google.genai import types as genai_types
import logging
import time
from typing import List, Optional
from .base import AIWrapper, AIResponse
from ..types import Message
from ..tools import Tool

class GeminiWrapper(AIWrapper):
    def __init__(self, api_key: str, model_name: str):
        """
        Wrapper for Google's Gemini API using the modern Google Gen AI SDK.
        
        Args:
            api_key: Google API key
            model_name: Gemini model name (e.g., 'gemini-2.0-flash-001')
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.client = genai.Client(api_key=api_key)
        
    def _messages(self, model_name: str, messages: List[Message], 
                 system_prompt: str, max_tokens: int, temperature: float,
                 tools: Optional[List[Tool]] = None,
                 thinking: bool = False, thinking_budget_tokens: Optional[int] = None) -> AIResponse:
        
        # Convert messages to the format expected by the new Google Gen AI SDK
        contents = []
        for message in messages:
            if message.role not in ["user", "assistant"]:
                self.logger.warning(f"Skipping message with unsupported role: {message.role}")
                continue
            
            parts = []
            for msg_content in message.content:
                if msg_content.type == "text":
                    parts.append(genai_types.Part.from_text(text=msg_content.text))
                elif msg_content.type == "image":
                    # For images, we need to use the proper format for the new SDK
                    parts.append(genai_types.Part.from_bytes(
                        data=msg_content.image["data"],
                        mime_type=msg_content.image["media_type"]
                    ))
            
            if not parts:
                self.logger.warning(f"Skipping message with empty content for role: {message.role}")
                continue
            
            # Map roles appropriately    
            role = "user" if message.role == "user" else "model"
            contents.append(genai_types.Content(role=role, parts=parts))
            
        if tools:
             self.logger.warning("Gemini tool usage not fully implemented in this basic wrapper.")
        
        # Create configuration with system instruction support
        config = genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
        )
        
        # Generate content using the new SDK
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        
        # Enhanced error handling for the new SDK
        if not hasattr(response, 'candidates') or not response.candidates:
            # Check for prompt-level blocking
            prompt_feedback = getattr(response, 'prompt_feedback', None)
            if prompt_feedback:
                block_reason = getattr(prompt_feedback, 'block_reason', None)
                safety_ratings = getattr(prompt_feedback, 'safety_ratings', [])
                
                if block_reason:
                    error_detail = f"Prompt was blocked due to {block_reason}. Safety ratings: {safety_ratings}"
                else:
                    error_detail = f"No candidates returned. Safety ratings: {safety_ratings}"
            else:
                error_detail = "No candidates returned and no prompt feedback available."
            
            raise Exception(error_detail)

        # Get the first candidate
        candidate = response.candidates[0]

        response_error = None
        
        # Check candidate's finish reason for various blocking scenarios
        finish_reason = getattr(candidate, 'finish_reason', None)
        if finish_reason:
            if finish_reason == "SAFETY":
                safety_ratings = getattr(candidate, 'safety_ratings', [])
                error_detail = f"Response was blocked due to safety concerns. Safety ratings: {safety_ratings}"
                raise Exception(error_detail)
            elif finish_reason == "RECITATION":
                error_detail = "Response was blocked due to recitation concerns. The output may resemble training data."
                raise Exception(error_detail)
            elif finish_reason in ["MAX_TOKENS", "OTHER"]:
                self.logger.warning(f"Response finished with reason: {finish_reason}")
                response_error = finish_reason
                # Continue processing as we might still have partial content
        
        # For the new SDK, use the .text property to get the response text
        try:
            response_text = response.text
            if response_text is None:
                print("RESPONSE")
                print(response)
                print("END RESPONSE")
            return AIResponse(content=response_text, error=response_error)
        except Exception as e:
            response_error = (response_error + " + " if response_error else "") + "Response does not have a .text property"
            # Fallback to manual extraction if .text property doesn't work
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    first_part = candidate.content.parts[0]
                    if hasattr(first_part, 'text') and first_part.text:
                        response_text = first_part.text
                        return AIResponse(content=response_text, error=response_error)
            
            raise Exception(f"Could not extract text from response: {e}")