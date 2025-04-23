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
        
        try:
            response = model.generate_content(
                gemini_messages, 
                generation_config=generation_config,
            )
                
            if not response.candidates:
                 prompt_feedback = getattr(response, 'prompt_feedback', None)
                 finish_reason = getattr(prompt_feedback, 'block_reason', 'Unknown') if prompt_feedback else 'Unknown'
                 safety_ratings = getattr(prompt_feedback, 'safety_ratings', []) if prompt_feedback else []
                 error_detail = f"No candidates returned. Finish reason: {finish_reason}. Safety: {safety_ratings}"
                 self.logger.error(error_detail)
                 return AIResponse(content=f"Error: {error_detail}") 
            
            response_text = response.candidates[0].content.parts[0].text
            return AIResponse(content=response_text)
                
        except Exception as e:
            self.logger.error(f"Error calling Gemini API: {str(e)}")
            return AIResponse(content=f"Error: {str(e)}") 