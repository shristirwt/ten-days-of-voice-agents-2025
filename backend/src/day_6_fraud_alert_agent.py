import logging
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class RazorpaySDR(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a warm, professional Sales Development Representative (SDR) for Razorpay.

YOU MUST FOLLOW THESE RULES:
1. START THE CONVERSATION: Immediately greet the user warmly.
   Example: "Hi! I'm your Razorpay assistant. What brings you here today? What are you working on?"

2. WHEN USERS ASK ABOUT RAZORPAY:
   - IMPORTANT: Always call the 'search_faq' function when user asks about product/pricing/features
   - Answer based on what the function returns
   - Be conversational

3. COLLECT LEAD INFORMATION:
   - IMPORTANT: You MUST call 'collect_lead_info' for each piece of information you get:
     - When they tell you their name, call: collect_lead_info("name", "their name")
     - When they tell you their company, call: collect_lead_info("company", "their company")
     - When they tell you their email, call: collect_lead_info("email", "their email")
     - When they tell you their role, call: collect_lead_info("role", "their role")
     - When they tell you use case, call: collect_lead_info("use_case", "their use case")
     - When they tell you team size, call: collect_lead_info("team_size", "their team size")
     - When they tell you timeline, call: collect_lead_info("timeline", "their timeline")

4. WHEN USER IS DONE:
   - IMPORTANT: Call the 'finalize_lead' function to generate summary and save the lead
   - Thank them warmly

KEY RULES:
- Be friendly and genuinely interested in their business
- Ask one question at a time
- If they ask something not in FAQ, be honest
- Keep it conversational
- ALWAYS call the appropriate function when relevant
"""
        )
        
        self.user_data = {
            "userName" : None,
                "securityIdentifier" : None,
                "cardEnding": None,
                "case": None,
                "transactionName" : None,
                "transactionTime" : None,
                "transactionCategory" : None,
                "transactionSource" : None
        }


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with the SDR agent
    await session.start(
        agent=RazorpaySDR(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
