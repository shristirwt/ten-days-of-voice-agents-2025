#!/bin/bash

# Start all services in background
livekit-server --dev &
(cd backend && uv run python src/day_6_fraud_alert_agent.py dev) &
(cd frontend && pnpm dev) &

# Wait for all background jobs
wait