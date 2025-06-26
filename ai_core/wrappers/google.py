import google.generativeai as genai
import logging
import time
from typing import List, Optional
from .base import AIWrapper, AIResponse
from ..types import Message
from ..tools import Tool

class GeminiWrapper(AIWrapper):
    def __init__(self, api_key: str, model_name: str):
        """
        Wrapper for Google's Gemini API.
        
        Args:
            api_key: Google API key
            model_name: Gemini model name (e.g., 'gemini-1.5-pro-latest')
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.logger = logging.getLogger(__name__)
        
    def _messages(self, model_name: str, messages: List[Message], 
                 system_prompt: str, max_tokens: int, temperature: float,
                 tools: Optional[List[Tool]] = None,
                 thinking: bool = False, thinking_budget_tokens: Optional[int] = None) -> AIResponse:
        
        model = self.model

        role_mapping = {"user": "user", "assistant": "model"}
        gemini_messages = []
        for message in messages:
            if message.role not in role_mapping:
                self.logger.warning(f"Skipping message with unsupported role: {message.role}")
                continue
            
            content = []
            for msg_content in message.content:
                if msg_content.type == "text":
                    content.append(msg_content.text)
                elif msg_content.type == "image":
                    content.append({
                        "mime_type": msg_content.image["media_type"],
                        "data": msg_content.image["data"]
                    })
            
            if not content:
                self.logger.warning(f"Skipping message with empty content for role: {message.role}")
                continue
                
            gemini_messages.append({
                "role": role_mapping.get(message.role),
                "parts": content
            })
            
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        
        system_instruction = None
        if system_prompt:
             system_instruction = genai.types.Content(
                 parts=[genai.types.Part(text=system_prompt)],
                 role="system"
             )
        
        if tools:
             self.logger.warning("Gemini tool usage not fully implemented in this basic wrapper.")
        
        response = model.generate_content(
            gemini_messages, 
            generation_config=generation_config,
        )
        
        # Enhanced error handling for no candidates
        if not response.candidates:
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
                # Continue processing as we might still have partial content
        
        # Check if content exists
        if not hasattr(candidate, 'content') or not candidate.content:
            error_detail = f"No content in candidate. Finish reason: {finish_reason}"
            raise Exception(error_detail)
        
        # Check if parts exist
        if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
            error_detail = f"No parts in candidate content. Finish reason: {finish_reason}"
            raise Exception(error_detail)
        
        # Check if the first part has text
        first_part = candidate.content.parts[0]
        if not hasattr(first_part, 'text') or not first_part.text:
            error_detail = f"No text in first part. Part type: {type(first_part)}. Finish reason: {finish_reason}"
            raise Exception(error_detail)

        response_text = first_part.text
        return AIResponse(content=response_text)