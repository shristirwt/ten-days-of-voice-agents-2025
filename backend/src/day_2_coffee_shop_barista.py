import logging

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
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class FriendlyBarista(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly, high-energy barista at "Code & Coffee". 
                            Your only goal is to take a complete coffee order efficiently and confirm it.
                            
                            REQUIRED FIELDS (you MUST collect all 4):
                            1. Drink Type (e.g., Latte, Cappuccino, Americano)
                            2. Size (e.g., Small, Medium, Large)
                            3. Milk Type (e.g., Oat, Almond, Regular, None)
                            4. Customer Name

                            YOUR PROCESS:
                            1. Listen to the user's request.
                            2. IMMEDIATELY call the function 'update_order' with any fields they mentioned.
                            3. Check your memory: Do you have all 4 required fields?
                            4. If ANY are missing, ask for that ONE specific missing item using a conversational tone.
                            Examples:
                            - "Great! What size would you like for that Latte?"
                            - "And what type of milk do you prefer?"
                            - "Perfect! What's your name for the order?"
                            5. Repeat until all 4 required fields are complete.
                            6. Ask "Any extras?" (optional - syrup, extra shot, etc.) only ONCE.
                            7. Once all 4 required fields are collected, call 'finalize_order'.

                            CRITICAL RULES:
                            - You CANNOT finalize the order without: Drink Type, Size, Milk, Name.
                            - Do NOT ask for multiple fields at once (e.g., "What size and milk?").
                            - Do NOT suggest items they didn't request.
                            - Keep responses SHORT and CONVERSATIONAL (optimized for spoken word).
                            - Once you call finalize_order, confirm: "Your order is confirmed! [Recap order]"
                        """,
        )
        self.order_state = {
            "drinkType": None,
            "size": None,
            "milk": None,
            "extras": [],
            "name": None
        }

    @function_tool
    async def update_order(self, context:RunContext = None, drinkType:str = None, size:str = None, milk:str = None, name:str = None, extras:list = None):
        
        """Update the customer's coffee order with provided details.
        Call this when the customer mentions their drink, size, milk preference, or name.
        
        Arguments:
            drinkType: The type of coffee (e.g. , 'latte', 'cappuccino', 'americano')
            size: The size (e.g., 'small', 'medium', 'large')
            milk: The milk type (e.g., 'oat', 'almond', 'regular', 'none')
            name: The customer's name for the order
            extras: Any additions like 'extra shot', 'caramel syrup'
        """
        
        logger.info(f'BEFORE UPDATE - drinkType param: {drinkType}, size param: {size}, milk param: {milk}, name param: {name}')
        logger.info(f'BEFORE UPDATE - Current state: {self.order_state}')
        if drinkType:
            self.order_state['drinkType'] = drinkType
        if size:
            self.order_state['size'] = size
        if milk:
            self.order_state['milk'] = milk
        if name:
            self.order_state['name'] = name
        if extras:
            self.order_state['extras'].append(extras)
            
        logger.info(f'AFTER UPDATE - Current state: {self.order_state}')
            
        return f"Order updated: {self.order_state}"
    
    @function_tool
    async def finalize_order(self, context:RunContext = None):
        """Finalize and confirm the customer's coffee order.
    
            Call this ONLY when you have collected all 4 required fields:
            - Drink Type
            - Size
            - Milk Type
            - Customer Name
            
            This function will:
            1. Validate that all 4 required fields are present
            2. Return an error if any field is missing
            3. Confirm the complete order if valid
        """
        
        logger.info(f"FINALIZE CALLED - Current state: {self.order_state}")
        fields = ['drinkType', 'size', 'milk', 'name']
        missing = [field for field in fields if not self.order_state[field]]
        logger.info(f"FINALIZE - Missing fields: {missing}")
        
        if missing:
            missing_list = ','.join(missing)
            return f"Cannot finalize order. Missing: {missing_list}. Please provide these details."
        
        order_summary = (
            f'Order confirmed!' 
            f"1 {self.order_state['size'].capitalize()} {self.order_state['drinkType'].capitalize()} "
            f"with {self.order_state['milk'].capitalize()} milk "
            f"for {self.order_state['name'].capitalize()}"
        )
        
        if self.order_state['extras']:
            extras_text = ", ".join(self.order_state['extras'])
            order_summary += f" - Extras: {extras_text}"
    
        logger.info(f"Order finalized: {self.order_state}")
        
        return order_summary

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=FriendlyBarista(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
