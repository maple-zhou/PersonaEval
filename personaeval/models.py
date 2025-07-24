"""Model management and API calls for PersonaEval."""

import json
import random
import re
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json_repair
import openai
import requests
from pydantic import BaseModel

from .config import ModelConfig, ExperimentConfig


class APIResponse(BaseModel):
    """Standardized API response."""
    
    content: str
    reasoning_content: Optional[str] = None
    completion_tokens: int
    prompt_tokens: int
    cost: float


class ModelManager:
    """Manages model API calls and response parsing."""
    
    def __init__(self, model_config: ModelConfig, experiment_config: ExperimentConfig):
        self.model_config = model_config
        self.reasoning_models = experiment_config.reasoning_models or []
        self.experiment_config = experiment_config
    
    def call_api(
        self, 
        prompt: str, 
        model_name: str,
        options: List[str],
        ground_truth: str,
        temperature: float = 0.0,
        max_retries: int = 5,
        sleep_interval: float = 60
    ) -> Tuple[str, APIResponse, Dict[str, float], int]:
        """
        Call the model API and return the result.
        
        Returns:
            Tuple of (predicted_answer, api_response, probabilities, completion_tokens)
        """
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(max_retries):
            try:
                if model_name in self.reasoning_models:
                    return self._call_api_stream(
                        messages, model_name, options, ground_truth, temperature
                    )
                else:
                    return self._call_api_standard(
                        messages, model_name, options, ground_truth, temperature
                    )
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(sleep_interval)
        
        raise Exception("All API call attempts failed")
    
    def _call_api_standard(
        self, 
        messages: List[Dict[str, str]], 
        model_name: str,
        options: List[str],
        ground_truth: str,
        temperature: float
    ) -> Tuple[str, APIResponse, Dict[str, float], int]:
        """Call API using standard completion endpoint."""
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        
        headers = {
            "Authorization": f"Bearer {self.model_config.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add proxy if configured
        proxies = None
        if self.model_config.proxy_url:
            proxies = {
                'http': self.model_config.proxy_url,
                'https': self.model_config.proxy_url
            }
        
        response = requests.post(
            self.model_config.url, 
            json=payload, 
            headers=headers, 
            timeout=self.experiment_config.timeout,
            proxies=proxies
        ).json()
        
        if "error" in response:
            raise Exception(response["error"])
        
        content = response["choices"][0]["message"]["content"]
        usage = response["usage"]
        
        # Calculate cost
        cost = (
            usage["prompt_tokens"] * self.model_config.cost_input + 
            usage["completion_tokens"] * self.model_config.cost_output
        )
        
        # Parse response and get probabilities
        probabilities = self._parse_response(content, options)
        predicted_answer = self._get_prediction(probabilities, ground_truth)
        
        api_response = APIResponse(
            content=content,
            completion_tokens=usage["completion_tokens"],
            prompt_tokens=usage["prompt_tokens"],
            cost=cost
        )
        
        return predicted_answer, api_response, probabilities, usage["completion_tokens"]
    
    def _call_api_stream(
        self, 
        messages: List[Dict[str, str]], 
        model_name: str,
        options: List[str],
        ground_truth: str,
        temperature: float
    ) -> Tuple[str, APIResponse, Dict[str, float], int]:
        """Call API using streaming endpoint for reasoning models."""

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {
                "include_usage": True
            }
        }

        headers = {
            "Authorization": f"Bearer {self.model_config.api_key}",
            "Content-Type": "application/json"
        }

        # Add proxy if configured
        proxies = None
        if self.model_config.proxy_url:
            proxies = {
                'http': self.model_config.proxy_url,
                'https': self.model_config.proxy_url
            }

        response = requests.post(
            self.model_config.url, 
            json=payload, 
            headers=headers, 
            stream=True, 
            timeout=self.experiment_config.timeout,
            proxies=proxies
        )

        if response.status_code != 200:
            print(response.json())
            raise Exception(response.json()["error"])

        reasoning_content = ""
        content = ""
        usage_info = None

        for line in response.iter_lines():
            if line:
                if line.startswith(b"data: "):
                    line = line[6:]

                if line == b"[DONE]":
                    break

                try:
                    chunk = json.loads(line.decode('utf-8'))

                    if "usage" in chunk:
                        usage_info = chunk["usage"]
                    
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            reasoning_content += delta["reasoning_content"]
                        elif "content" in delta and delta["content"] is not None:
                            content += delta["content"]
                except json.JSONDecodeError:
                    continue
        
        # Calculate cost
        cost = (
            usage_info["prompt_tokens"] * self.model_config.cost_input + 
            usage_info["completion_tokens"] * self.model_config.cost_output
        )
        
        # Parse response and get probabilities
        probabilities = self._parse_response(content, options)
        predicted_answer = self._get_prediction(probabilities, ground_truth)
        
        api_response = APIResponse(
            content=content,
            reasoning_content=reasoning_content,
            completion_tokens=usage_info["completion_tokens"],
            prompt_tokens=usage_info["prompt_tokens"],
            cost=cost
        )
        
        return predicted_answer, api_response, probabilities, usage_info["completion_tokens"]
    
    def _parse_response(self, response: str, options: List[str]) -> Dict[str, float]:
        """Parse model response to extract probabilities for each option."""
        
        # Try to extract JSON from code blocks first
        pattern = r"```\s*(.+?)\s*```"
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            json_str = matches[-1]  # Take the last match
        else:
            json_str = response
        
        try:
            parsed_obj = json_repair.loads(json_str)
            
            # Validate that all options are present
            for option in options:
                if option not in parsed_obj:
                    raise ValueError(f"Option '{option}' not found in response")
            
            # Convert to float and validate
            result = {}
            for option in options:
                result[option] = float(parsed_obj[option])
            
            # Check if probabilities sum to 1 (with some tolerance)
            total = sum(result.values())
            if abs(total - 1.0) > 1e-5:
                raise ValueError(f"Probabilities do not sum to 1: {total}")
            
            return result
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response: {response}")
            raise e
    
    def _get_prediction(
        self, 
        probabilities: Dict[str, float], 
        ground_truth: str
    ) -> str:
        """Get the predicted answer from probabilities."""

        return max(probabilities, key=probabilities.get)