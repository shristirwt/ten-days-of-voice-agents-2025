#!/bin/bash

# Start all services in background
livekit-server --dev &
(cd backend && uv run python src/day_3_wellness_agent.py dev) &
(cd frontend && pnpm dev) &

# Wait for all background jobs
wait