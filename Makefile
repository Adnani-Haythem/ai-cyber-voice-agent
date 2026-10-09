PYTHON ?= .venv/bin/python
APP_MODULE = app.main
OLLAMA_MODEL ?= llama3.2

.PHONY: run install

run:
	@set -eu; \
	ollama_pid=""; \
	cleanup() { \
		if [ -n "$$ollama_pid" ] && kill -0 "$$ollama_pid" 2>/dev/null; then \
			kill "$$ollama_pid"; \
		fi; \
	}; \
	trap cleanup EXIT INT TERM; \
	if ollama list >/dev/null 2>&1; then \
		echo "Ollama is already running."; \
	else \
		echo "Starting Ollama..."; \
		ollama serve >/tmp/ai-cyber-voice-agent-ollama.log 2>&1 & \
		ollama_pid=$$!; \
		until ollama list >/dev/null 2>&1; do \
			if ! kill -0 "$$ollama_pid" 2>/dev/null; then \
				cat /tmp/ai-cyber-voice-agent-ollama.log; \
				exit 1; \
			fi; \
			sleep 1; \
		done; \
	fi; \
	if ! ollama list | awk 'NR > 1 {print $$1}' | grep -Fxq '$(OLLAMA_MODEL):latest'; then \
		echo "Downloading Ollama model $(OLLAMA_MODEL)..."; \
		ollama pull $(OLLAMA_MODEL); \
	fi; \
	FRAUD_MONITOR_ENABLED=1 $(PYTHON) -m $(APP_MODULE)

install:
	$(PYTHON) -m pip install -r requirements.txt
