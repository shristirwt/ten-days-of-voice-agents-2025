import logging
import json
from pathlib import Path

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


class TeachTheTutor(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are an intelligent tutoring coach with three powerful learning modes.
            Your goal is to help users master concepts through explanation, questioning, and active recall.

            IMPORTANT: You are NOT a medical or clinical professional. Focus only on educational concepts.

            THE THREE LEARNING MODES:

            1. LEARN MODE (Agent explains)
            - Call the 'explain_concept' function with the concept the user asks about
            - Explain clearly, break ideas into simple parts
            - Use analogies and examples
            - After explaining, ask: "Ready to test yourself with a quiz?"

            2. QUIZ MODE (Agent asks questions)
            - Call the 'ask_question' function with the concept
            - Ask the quiz question
            - Listen to the user's answer
            - Give immediate feedback: "Good!" or "Close, but..." or "Not quite..."
            - Offer: "Want another question or try teaching it back?"

            3. TEACH_BACK MODE (User teaches, agent scores)
            - Call the 'score_explanation' function
            - Say: "Now it's your turn! Explain [concept] to me as if I'm a beginner."
            - Listen carefully to their explanation
            - Score based on whether they cover key ideas
            - Give constructive feedback: what they got right, what they missed

            YOUR PROCESS:
            1. Greet warmly: "Hi! I'm your learning tutor. What would you like to do?"
            2. Ask for mode: "Choose LEARN (I explain), QUIZ (answer my questions), or TEACH_BACK (you explain)"
            3. Ask for concept: "Which concept? (Variables, Loops, Functions, etc.)"
            4. Call the appropriate function tool
            5. After each mode, ask: "Want to switch modes, try another concept, or keep going?"

            CRITICAL RULES:
            - Be encouraging and supportive, never judgmental
            - Ask questions one at a time
            - Keep explanations SHORT and conversational (optimized for spoken word)
            - Use simple language - assume learner is a beginner
            - In TEACH_BACK mode, let them explain fully before giving feedback
            - If user asks to switch modes, switch immediately - no resistance
            - If user asks about a concept not in your content, apologize and suggest available ones
            - Always celebrate progress: "Great job!", "You're learning fast!", "That's excellent!"

            FEEDBACK STYLE:
            - LEARN: Clear, concise explanations with examples
            - QUIZ: Encouraging feedback, hint if they struggle
            - TEACH_BACK: Acknowledge what they got right first, then gently point out gaps

            Remember: Active recall (teaching back) is the most powerful learning tool!
            """,
        )
        
        # Hardcoded learning concepts
        self.concepts = [
            {
                "id": "variables",
                "title": "Variables",
                "summary": "Variables are containers that store data values. Think of them like labeled boxes where you can put information and use it later. Variables allow you to reuse values without typing them repeatedly. You give each variable a name, assign a value to it, and then reference it by name whenever you need that value.",
                "sample_question": "What is a variable and why is it useful?"
            },
            {
                "id": "loops",
                "title": "Loops",
                "summary": "Loops are programming structures that let you repeat an action or block of code multiple times automatically. Instead of writing the same code over and over, you write it once and tell the loop how many times to execute it or until a condition is met. There are different types of loops like for loops and while loops.",
                "sample_question": "Explain the difference between a for loop and a while loop."
            },
            {
                "id": "functions",
                "title": "Functions",
                "summary": "Functions are reusable blocks of code that perform a specific task. Instead of writing the same logic repeatedly throughout your program, you write it once in a function and call it whenever you need it. Functions can take inputs called parameters and return outputs called return values.",
                "sample_question": "What is a function and how does it help you write better code?"
            },
            {
                "id": "conditionals",
                "title": "Conditionals",
                "summary": "Conditionals are statements that allow your program to make decisions. They check if something is true or false and execute different code based on the result. Common conditional statements are if, else if, and else. They help your program behave differently depending on the data it receives.",
                "sample_question": "What are conditional statements and give an example of when you would use an if-else statement."
            },
            {
                "id": "arrays",
                "title": "Arrays",
                "summary": "Arrays are data structures that store multiple values of the same type in a single variable. Instead of creating separate variables for each item, you can group related items in an array. You access each item using an index number, starting from zero. Arrays are useful for storing lists of data like students, scores, or product names.",
                "sample_question": "What is an array and how do you access a specific element in an array?"
            }
        ]
        logger.info(f"Loaded {len(self.concepts)} concepts")
            
    @function_tool
    async def explain_concept(self, concept_id: str):
        """Explain a learning concept in detail using simple language.

        Call this function when the user asks to learn about a concept or chooses LEARN mode.
        This function loads the concept from the content file and provides a clear, beginner-friendly explanation.
        
        Args:
            id: The ID of the concept to explain (e.g., 'variables', 'loops', 'functions')
                    Examples: 'variables', 'loops', 'functions', 'conditionals'
        
        Returns:
            A clear, detailed explanation of the concept with examples and analogies.
        """
        
        concept = self._find_concept(concept_id)
    
        if not concept:
            available = ", ".join([c['id'] for c in self.concepts])
            return f"Sorry, I don't know about '{concept_id}'. I can teach: {available}"
        
        # Get title and explanation from JSON
        title = concept['title']
        explanation = concept['summary']
        
        logger.info(f"Explaining concept: {title}")
        
        return f"Great! Let me explain {title}:\n{explanation}"
    
    def _find_concept(self, concept_id: str):
        """Find a concept by its ID"""
        if not concept_id:
            logger.warning("concept_id is empty or None")
            return None
        
        logger.info(f"Searching for concept: '{concept_id}' (lowercased: '{concept_id.lower()}')")
        logger.info(f"Available concepts: {[c.get('id', '?') for c in self.concepts]}")
        
        for concept in self.concepts:
            if concept['id'].lower() == concept_id.lower():
                logger.info(f"Found concept: {concept['id']}")
                return concept
        
        logger.warning(f"Concept '{concept_id}' not found in loaded concepts")
        return None
    
    @function_tool
    async def ask_question(self, concept_id: str):
        """Ask a quiz question about a learning concept.
    
        Call this function when the user chooses QUIZ mode or wants to test their knowledge.
        This pulls a question from the content file and asks the user to answer it.
        Listen to their response and provide immediate feedback.
        
        Args:
            concept_id: The ID of the concept to quiz on (e.g., 'variables', 'loops', 'functions')
                    Examples: 'variables', 'loops', 'functions', 'conditionals'
        
        Returns:
            A clear, focused quiz question that tests understanding of the concept.
            Example: "What is a variable and why is it useful?"
        """
        concept = self._find_concept(concept_id)
    
        if not concept:
            available = ", ".join([c['id'] for c in self.concepts])
            return f"Sorry, I don't know about '{concept_id}'. Available concepts: {available}"
        
        question = concept['sample_question']
        title = concept['title']
        
        logger.info(f"Asking question about: {title}")
        
        return f"Here's your quiz question about {title}: {question}"
        
    @function_tool
    async def score_explanation(self, concept_id: str, user_explanation: str):
        """Score and provide feedback on the user's explanation of a concept.
    
        Call this function when the user chooses TEACH_BACK mode and explains a concept to you.
        Evaluate their explanation based on key concepts and provide constructive, encouraging feedback.
        
        Args:
            concept_id: The ID of the concept being explained (e.g., 'variables', 'loops', 'functions')
            user_explanation: The user's explanation of the concept in their own words.
                            This is what they said when you asked them to teach the concept.
        
        Returns:
            Constructive feedback that:
            - Acknowledges what they explained correctly
            - Points out what they missed or misunderstood (if anything)
            - Provides encouragement and next steps
            
            Examples of feedback:
            - "Excellent! You clearly explained that variables store values and can be reused. You also mentioned naming conventions - great thinking!"
            - "Good start! You're right that loops repeat actions. But remember, loops need a stopping condition or they run forever."
            - "I appreciate the effort! Variables aren't actually functions - they're containers for data. Think of them as labeled boxes..."
        """
        
        concept = self._find_concept(concept_id)
    
        if not concept:
            available = ", ".join([c['id'] for c in self.concepts])
            return f"Sorry, I don't know about '{concept_id}'. Available concepts: {available}"
        
        if not user_explanation:
            return "I didn't hear your explanation. Could you please explain the concept to me?"
        
        # Get key information
        title = concept['title']
        summary = concept['summary']
        
        # Extract key words/concepts from the summary
        key_concepts = self._extract_key_concepts(summary)
        
        # Check how many key concepts the user mentioned
        user_explanation_lower = user_explanation.lower()
        mentioned_concepts = []
        missing_concepts = []
        
        for key in key_concepts:
            if key.lower() in user_explanation_lower:
                mentioned_concepts.append(key)
            else:
                missing_concepts.append(key)
        
        # Calculate score percentage
        score_percentage = (len(mentioned_concepts) / len(key_concepts)) * 100 if key_concepts else 0
    
        logger.info(f"Score for {title}: {score_percentage}% - Mentioned: {mentioned_concepts}, Missing: {missing_concepts}")
        
        # Build feedback based on score
        if score_percentage >= 80:
            # Excellent explanation
            feedback = f"Excellent! You explained {title} really well! You covered: {', '.join(mentioned_concepts)}. "
            if missing_concepts:
                feedback += f"One small thing you could add is: {', '.join(missing_concepts[:1])}. "
            feedback += "You're mastering this concept!"
            
        elif score_percentage >= 50:
            # Good start but missing some parts
            feedback = f"Good job! You got the main idea right. You mentioned: {', '.join(mentioned_concepts)}. "
            feedback += f"But remember, {title} also involves: {', '.join(missing_concepts[:2])}. "
            feedback += "You're on the right track!"
            
        else:
            # Needs more work
            feedback = f"I appreciate the effort! You mentioned some good points: {', '.join(mentioned_concepts) if mentioned_concepts else 'a few ideas'}. "
            feedback += f"But {title} is really about: {', '.join(missing_concepts[:3])}. "
            feedback += "Let me re-explain this, and you can try again!"
        
        return feedback

    def _extract_key_concepts(self, summary: str) -> list:
        """Extract key concepts from the concept summary.
        
        This is a simple implementation - you can make it more sophisticated.
        It splits by common keywords and returns important terms.
        """
        
        # Simple approach: split by common delimiters and take key words
        # In a real system, you might use NLP here
        
        # Remove common words
        stop_words = {'a', 'an', 'the', 'is', 'are', 'be', 'to', 'and', 'or', 'in', 'on', 'at', 'by', 'for', 'of', 'with', 'you', 'can', 'so', 'if', 'that', 'it', 'as'}
        
        # Split into words
        words = summary.lower().split()
        
        # Filter: keep words longer than 3 characters and not in stop words
        key_words = [w.strip('.,!?') for w in words if len(w) > 3 and w.lower() not in stop_words]
        
        # Remove duplicates and return first 5 key concepts
        return list(dict.fromkeys(key_words))[:5]

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
        agent=TeachTheTutor(),
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
