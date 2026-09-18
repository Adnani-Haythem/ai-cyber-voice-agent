"""
Action Executor - Executes security actions
"""

import re
from typing import Dict, Any, List

class ActionExecutor:
    """Execute security actions"""

    def __init__(self, log_file="./data/logs/secure.log"):
        self.log_file = log_file

    def execute(self, intent: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Execute the intent"""
        action = intent.get("action", "")
        params = intent.get("parameters", {})

        print(f"⚡ Executing action: {action} with params: {params}")

        if action == "QUERY_LOGS":
            return self._query_logs(params, context)
        elif action == "ENRICH_IP":
            return self._enrich_ip(params)
        elif action == "ISOLATE_HOST":
            return self._isolate_host(params)
        elif action == "BLOCK_IP":
            return self._block_ip(params)
        elif action == "SUMMARIZE":
            return self._summarize(params, context)
        else:
            return {"summary": f"Unknown action: {action}", "status": "error"}

    def _query_logs(self, params: Dict, context: str) -> Dict[str, Any]:
        """Query logs using RAG context"""
        if context:
            lines = context.split('\n')
            return {
                "summary": f"Found {len(lines)} relevant log entries",
                "data": lines[:5],
                "status": "success"
            }

        return {"summary": "No relevant logs found", "data": [], "status": "success"}

    def _enrich_ip(self, params: Dict) -> Dict[str, Any]:
        ip = params.get("ip", "unknown")
        return {
            "summary": f"IP {ip} - Threat Level: HIGH, Reputation: Suspicious",
            "status": "success"
        }

    def _isolate_host(self, params: Dict) -> Dict[str, Any]:
        ip = params.get("ip", "unknown")
        return {
            "summary": f"Host {ip} ISOLATED. Incident: INC-{hash(ip) % 10000}",
            "status": "success"
        }

    def _block_ip(self, params: Dict) -> Dict[str, Any]:
        ip = params.get("ip", "unknown")
        return {
            "summary": f"IP {ip} BLOCKED. Rule: RULE-{hash(ip) % 5000}",
            "status": "success"
        }

    def _summarize(self, params: Dict, context: str) -> Dict[str, Any]:
        if context:
            return {"summary": f"Summary: {context[:200]}...", "status": "success"}

        return {"summary": "No logs available", "status": "success"}
