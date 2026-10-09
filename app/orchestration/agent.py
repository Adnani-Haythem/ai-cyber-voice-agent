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

    def process(self, transcript: str, approval=None) -> str:
        """Main processing pipeline"""
        print(f"🤖 Processing: {transcript}")

        # Step 1: Retrieve context (RAG)
        context_chunks = self.rag_engine.retrieve(transcript, k=3)
        context = "\n".join(context_chunks) if context_chunks else ""

        # Step 2: Parse intent (LLM)
        intent = self.intent_parser.parse_intent(transcript, context)
        print(f"🎯 Intent: {json.dumps(intent, indent=2)}")

        # Step 3: Check if approval needed
        high_risk_actions = {"ISOLATE_HOST", "BLOCK_IP"}
        requires_approval = (
            intent.get("requires_approval", False)
            or intent.get("action") in high_risk_actions
        )
        if requires_approval:
            print(f"🔐 Approval required for: {intent['action']}")
            if approval is None:
                return self._approval_request(intent)
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

    def _approval_request(self, intent: Dict) -> str:
        """Return a browser-facing approval request without blocking the server."""
        action = intent.get("action", "unknown")
        params = intent.get("parameters", {})
        return (
            f"Approval required before {action} for "
            f"{params.get('ip', 'the requested target')}."
        )

    def _generate_response(self, intent: Dict, result: Dict) -> str:
        """Generate natural language response"""
        action = intent.get("action", "")

        if action == "QUERY_LOGS":
            data = result.get("data", [])
            if not data:
                return "Found 0 relevant log entries."
            lines = "\n".join(f"- {line}" for line in data)
            return f"Found {len(data)} relevant log entries:\n{lines}"
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
