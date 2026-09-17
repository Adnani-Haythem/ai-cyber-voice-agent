"""
LLM-based Intent Parser
Converts natural language to structured JSON actions
"""

import json
import os
import re
from typing import Dict, Any, Optional

class IntentParser:
    """Parse user intent using LLM (OpenAI or local)"""

    def __init__(self, use_openai: bool = True, use_local: bool = False, model_name: str = "llama3.2"):
        self.use_openai = use_openai
        self.use_local = use_local
        self.model_name = model_name
        self.client = None

        if use_openai:
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = OpenAI(api_key=api_key)
                    print("✅ OpenAI client initialized")
                else:
                    print("⚠️ No OpenAI API key found")
                    self.use_openai = False
            except ImportError:
                print("⚠️ OpenAI not installed")
                self.use_openai = False

        if use_local:
            try:
                import ollama
                self.ollama = ollama
                print(f"✅ Ollama client initialized with model: {model_name}")
            except ImportError:
                print("⚠️ Ollama not installed. Install with: pip install ollama")
                self.use_local = False

        if not self.use_openai and not self.use_local:
            print("⚠️ No LLM available, using fallback parser")

    def parse_intent(self, transcript: str, context: str = "") -> Dict[str, Any]:
        """Parse user command into structured JSON"""

        if self.use_openai and self.client:
            return self._parse_with_openai(transcript, context)
        elif self.use_local:
            return self._parse_with_ollama(transcript, context)
        else:
            return self._parse_with_rules(transcript)

    def _parse_with_openai(self, transcript: str, context: str) -> Dict[str, Any]:
        """Use OpenAI to parse intent"""
        try:
            system_prompt = """You are a cybersecurity intent parser. Convert user commands to JSON.

Available actions:
- QUERY_LOGS: Search/find logs
- SUMMARIZE: Activity summaries
- ENRICH_IP: IP threat intelligence
- ISOLATE_HOST: Isolate host (requires approval)
- BLOCK_IP: Block IP (requires approval)

Return JSON:
{
  "action": "ACTION_NAME",
  "parameters": {"param": "value"},
  "requires_approval": true/false,
  "confidence": 0.9,
  "explanation": "Brief explanation"
}"""

            user_prompt = f"""
Context from logs:
{context[:500] if context else "No context"}

User command: "{transcript}"

Parse the intent and return JSON only.
"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"❌ OpenAI parsing error: {e}")
            return self._parse_with_rules(transcript)

    def _parse_with_ollama(self, transcript: str, context: str) -> Dict[str, Any]:
        """Use local Ollama to parse intent"""
        try:
            system_prompt = """You are a cybersecurity intent parser. Convert user commands to JSON.

Available actions:
- QUERY_LOGS: Search/find logs
- SUMMARIZE: Activity summaries
- ENRICH_IP: IP threat intelligence
- ISOLATE_HOST: Isolate host (requires approval)
- BLOCK_IP: Block IP (requires approval)

Return JSON:
{
  "action": "ACTION_NAME",
  "parameters": {"param": "value"},
  "requires_approval": true/false,
  "confidence": 0.9,
  "explanation": "Brief explanation"
}

Return ONLY the JSON, no other text."""

            user_prompt = f"""
Context from logs:
{context[:500] if context else "No context"}

User command: "{transcript}"

Parse the intent and return JSON only.
"""

            response = self.ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False
            )

            # Extract the response text
            content = response['message']['content']

            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result

            # If no JSON found, try parsing the whole response
            try:
                result = json.loads(content)
                return result
            except:
                print(f"⚠️ Could not parse Ollama response as JSON: {content[:100]}...")
                return self._parse_with_rules(transcript)

        except Exception as e:
            print(f"❌ Ollama parsing error: {e}")
            return self._parse_with_rules(transcript)

    def _parse_with_rules(self, transcript: str) -> Dict[str, Any]:
        """Fallback rule-based parser"""
        transcript_lower = transcript.lower()

        ip_match = re.search(r'\d+\.\d+\.\d+\.\d+', transcript)
        ip = ip_match.group(0) if ip_match else None

        # Check for amount
        amount_match = re.search(r'\b(\d+[\.,]?\d*)\b', transcript)
        amount = float(amount_match.group(1).replace(",", "")) if amount_match else None

        if "isolate" in transcript_lower:
            return {
                "action": "ISOLATE_HOST",
                "parameters": {"ip": ip or "unknown"},
                "requires_approval": True,
                "confidence": 0.8,
                "explanation": "Host isolation requested"
            }
        elif "block" in transcript_lower:
            return {
                "action": "BLOCK_IP",
                "parameters": {"ip": ip or "unknown"},
                "requires_approval": True,
                "confidence": 0.8,
                "explanation": "IP blocking requested"
            }
        elif "enrich" in transcript_lower:
            return {
                "action": "ENRICH_IP",
                "parameters": {"ip": ip or "unknown"},
                "requires_approval": False,
                "confidence": 0.9,
                "explanation": "IP enrichment requested"
            }
        elif "summary" in transcript_lower or "summarize" in transcript_lower:
            return {
                "action": "SUMMARIZE",
                "parameters": {"timeframe": "24 hours"},
                "requires_approval": False,
                "confidence": 0.9,
                "explanation": "Activity summary requested"
            }
        elif "failed" in transcript_lower:
            return {
                "action": "QUERY_LOGS",
                "parameters": {"query": transcript},
                "requires_approval": False,
                "confidence": 0.8,
                "explanation": "Log query requested"
            }
        elif "check transaction" in transcript_lower or "fraud" in transcript_lower:
            return {
                "action": "CHECK_FRAUD",
                "parameters": {"amount": amount or 0, "transaction_type": 1},
                "requires_approval": False,
                "confidence": 0.8,
                "explanation": "Fraud check requested"
            }
        else:
            return {
                "action": "UNKNOWN",
                "parameters": {},
                "requires_approval": False,
                "confidence": 0.3,
                "explanation": "Unknown command"
            }
