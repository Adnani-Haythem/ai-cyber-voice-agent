"""
AI Agent Orchestration
Coordinates RAG, LLM, and Actions
"""

import json
from typing import Dict, Any, List

class SecurityAgent:
    """Orchestrates the AI security agent workflow"""

    def __init__(self, rag_engine, intent_parser, action_executor):
        self.rag_engine = rag_engine
        self.intent_parser = intent_parser
        self.action_executor = action_executor
        self.history = []
        self.pending_approvals = []

    def process(self, transcript: str) -> str:
        """Main processing pipeline"""
        print(f"🤖 Processing: {transcript}")

        # Step 1: Retrieve context (RAG)
        context_chunks = self.rag_engine.retrieve(transcript, k=3)
        context = "\n".join(context_chunks) if context_chunks else ""

        # Step 2: Parse intent (LLM)
        intent = self.intent_parser.parse_intent(transcript, context)
        print(f"🎯 Intent: {json.dumps(intent, indent=2)}")

        # Step 3: Check if approval needed
        if intent.get("requires_approval", False):
            print(f"🔐 Approval required for: {intent['action']}")
            approval = self._request_approval(intent)
            if not approval:
                return "⛔ Action cancelled. Approval denied."

        # Step 4: Execute action
        result = self.action_executor.execute(intent, context)
        print(f"✅ Result: {result}")

        # Step 5: Generate response
        response = self._generate_response(intent, result)

        # Log to history
        self.history.append({
            "transcript": transcript,
            "intent": intent,
            "result": result,
            "response": response
        })

        return response

    def _request_approval(self, intent: Dict) -> bool:
        """Request approval for high-risk actions"""
        action = intent.get("action", "unknown")
        params = intent.get("parameters", {})

        print("\n" + "="*50)
        print(f"🔐 APPROVAL REQUIRED")
        print(f"Action: {action}")
        print(f"Parameters: {json.dumps(params, indent=2)}")
        print("="*50)

        # In production, this would send a notification
        # For demo, we'll ask via terminal
        response = input("Approve this action? (yes/no): ").strip().lower()

        return response in ["yes", "y", "approve"]

    def _generate_response(self, intent: Dict, result: Dict) -> str:
        """Generate natural language response"""
        action = intent.get("action", "")

        if action == "QUERY_LOGS":
            data = result.get("data", [])
            return f"Found {len(data)} relevant log entries."
        elif action == "ISOLATE_HOST":
            return f"⚠️ Host {intent.get('parameters', {}).get('ip', 'unknown')} ISOLATED. Action logged."
        elif action == "BLOCK_IP":
            return f"🛑 IP {intent.get('parameters', {}).get('ip', 'unknown')} BLOCKED. Action logged."
        elif action == "ENRICH_IP":
            return result.get("summary", "IP enrichment completed")
        elif action == "SUMMARIZE":
            return result.get("summary", "Summary generated")
        else:
            return result.get("summary", "Action completed")
