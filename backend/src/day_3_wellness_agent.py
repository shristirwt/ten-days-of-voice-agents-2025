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


class WellnessAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a warm, supportive health and wellness companion. Your role is to check in   with the user about their day through a caring, non-judgmental conversation.

            IMPORTANT: You are NOT a medical professional. Never diagnose, prescribe, or make medical claims. Your job is to listen, understand, and offer simple, grounded support.

            YOUR PROCESS:
            1. Greet the user warmly and check how they're feeling today.
            2. Ask about their mood and energy levels in a conversational way.
            Examples of questions (vary them, don't repeat):
            - "How are you feeling today?"
            - "How's your energy level right now?"
            - "Anything on your mind that's stressing you out?"
            3. Ask about their goals and intentions for today.
            - "What are 1-3 things you'd like to accomplish today?"
            - "Is there anything you want to do for yourself today - rest, exercise, hobbies?"
            4. Based on what they share, offer simple, actionable advice or reflections.
            - Break large goals into smaller, manageable steps.
            - Suggest taking short breaks.
            - Offer grounding ideas like "take a 5-minute walk" or "drink some water."
            5. Recap the conversation back to them:
            - Summarize their mood
            - List their main 1-3 objectives
            - Ask: "Does this sound right?"
            6. Once confirmed, call the 'finalize_checkin' function to save everything.

            CRITICAL RULES:
            - Be warm and conversational, not robotic.
            - Ask questions one at a time, naturally.
            - NEVER diagnose, suggest treatment, or make medical claims.
            - NEVER ask for personal health information beyond mood/energy.
            - Keep responses SHORT and natural (optimized for spoken word).
            - Reference previous check-ins if available (e.g., "Last time you mentioned being low on energy. How's that today?").
            - Use non-judgmental language - this is a supportive space.
            - Once you have mood and objectives (at least 1), offer reflection/advice, then recap and finalize.
                        """,
        )
        
        self.user_mood = {
            'mood' : None,
            'energy' : None,
            'objectives' : []
        }

    @function_tool
    async def capture_mood(self, context:RunContext = None, mood:str = None, energy:str = None):
        """Capture the user's current mood and energy level for today's check-in.
    
            Call this function when the user shares how they're feeling emotionally or their energy level.
            Store their self-reported mood and energy in your memory.
            
            Args:
                mood: The user's emotional state or how they're feeling (e.g., 'good', 'stressed', 'anxious', 'calm', 'tired but motivated')
                energy: The user's energy level (e.g., 'high', 'medium', 'low', 'exhausted', 'energized')
            
            Returns:
                Confirmation of what was captured and encourages them to share their goals next.
        """
        if mood:
            self.user_mood['mood'] = mood
        if energy:
            self.user_mood['energy'] = energy
            
        return f"User mood updated: {self.user_mood}"
    
    @function_tool
    async def capture_objectives(self, context: RunContext = None, objectives:list = None):
        """Capture the user's goals and objectives for today.
    
            Call this function when the user shares what they want to accomplish or do for themselves today.
            Store their 1-3 main goals/objectives in your memory.
            
            Args:
                objectives: A list of 1-3 things the user wants to do today (e.g., ['finish the report', 'take a walk', 'call a friend'])
                These should be practical, achievable goals - not vague wishes.
            
            Returns:
                Confirmation of what was captured and prepares to offer advice/reflection.
        """
        if objectives:
            self.user_mood['objectives']= objectives
            
        
        objectives_str = ", ".join(objectives) if objectives else "nothing yet"
        return f"Got it! Your goals for today: {objectives_str}. Let me offer you some thoughts on this."
        
    
    @function_tool
    async def finalize_checkin(self, context: RunContext = None):
        """Finalize today's check-in, provide a recap, and save everything to the wellness log.
    
            Call this function ONLY after you have:
            1. Captured the user's mood and energy
            2. Captured their objectives for today
            3. Offered simple, grounded advice or reflection
            4. Recapped the conversation back to them for confirmation
            
            This function will:
            - Generate a brief summary of the check-in
            - Save the mood, objectives, and summary to wellness_log.json
            - Include date/time of the check-in
            - Return a closing message to end the session warmly
            
            Returns:
                A warm closing confirmation that the check-in was saved and encouragement for their day.
        """
        
        fields = ['mood', 'energy', 'objectives']
        missing = [field for field in fields if not self.user_mood[field]]
        
        if missing:
            missing_list = ",".join(missing)
            return f"Cannot finalize checkin. Missing: {missing_list}. Please provide these details."
        
        advice = self._generate_advice()
        
        checkin_summary = (
            f"You are feeling {self.user_mood['mood'].capitalize()} "
            f"Your energy is {self.user_mood['energy'].capitalize()} "
            f"Your objectives for today are: {','.join(self.user_mood['objectives'])} "
            f"Here's my thought: {advice}"
        )
        
        self._save_to_json()
        
        return checkin_summary
    
    def _generate_advice(self):
        """Generate grounded, actionable advice based on mood and objectives."""
        
        mood = self.user_mood['mood'].lower()
        energy = self.user_mood['energy'].lower()
        objectives = self.user_mood['objectives']
        
        advice_parts = []
        
        if 'low' in energy or 'tired' in energy:
            advice_parts.append("Since your energy is low, break your goals into smaller steps and take breaks between them.")
        elif 'high' in energy or 'energized' in energy:
            advice_parts.append("Great energy! Use this momentum, but remember to take short breaks to sustain it.")
        
        if 'stressed' in mood or 'anxious' in mood:
            advice_parts.append("Try a 5-minute grounding exercise - take deep breaths or a short walk to center yourself.")
        elif 'good' in mood or 'calm' in mood:
            advice_parts.append("You're in a good headspace. Channel this into your most important goal first.")
        
        if len(objectives) > 3:
            advice_parts.append("You have a lot on your plate - prioritize the 1-2 most important ones today.")
        elif len(objectives) == 1:
            advice_parts.append("Focusing on one goal is smart. Break it into smaller steps if needed.")
            
        if not advice_parts:
            advice_parts.append("Take it one step at a time and be kind to yourself today.")
    
        return " ".join(advice_parts)
    
    def _save_to_json(self):
        """Save the current check-in to wellness_log.json"""
    
        import json
        from datetime import datetime
        
        checkin_entry = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "mood": self.user_mood.get('mood', 'not specified'),
            "energy": self.user_mood.get('energy', 'not specified'),
            "objectives": self.user_mood.get('objectives', []),
            "summary": self._generate_advice()
        }
        
        log_file = "wellness_log.json"
        
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"checkins": []}
        
        data["checkins"].append(checkin_entry)
        
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Check-in saved to {log_file}")

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
        agent=WellnessAgent(),
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
