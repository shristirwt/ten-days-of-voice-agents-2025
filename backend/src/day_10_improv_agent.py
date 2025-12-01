import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    function_tool,
    cli,
    WorkerOptions,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


def load_scenarios():
    """Load improv scenarios from JSON file."""
    scenarios_path = Path(__file__).parent.parent / "shared-data" / "day_10_scenarios.json"
    logger.info(f"Loading scenarios from: {scenarios_path}")
    logger.info(f"Scenarios file exists: {scenarios_path.exists()}")
    
    default_scenarios = [
        {
            "round": 1,
            "scenario": "You are a barista who just realized that you've been serving coffee to the same customer for 20 years, and they still don't know your name. Tell them.",
        },
        {
            "round": 2,
            "scenario": "You're a museum tour guide, but you just found out that the famous painting you've been describing for years is actually a fake. How do you tell the tourists?",
        },
        {
            "round": 3,
            "scenario": "You're a weather forecaster who has to deliver news about an upcoming storm, but you're absolutely petrified of storms. Stay in character while warning people.",
        },
    ]
    
    try:
        with open(scenarios_path, 'r') as f:
            scenarios = json.load(f)
            logger.info(f"Successfully loaded {len(scenarios)} scenarios from file")
            return scenarios
    except Exception as e:
        logger.warning(f"Error loading scenarios from file: {e}. Using default scenarios.")
        return default_scenarios


class ImprovisationHost(Agent):
    """Improv battle game show host agent."""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are the host of a TV improv show called "Improv Battle". Your role is to run an exciting, entertaining improv game.

YOUR PERSONALITY:
- High-energy, witty, and clear about rules
- Sometimes amused, sometimes unimpressed, sometimes pleasantly surprised
- Not always supportive; light teasing and honest critique are allowed
- Stay respectful and non-abusive
- Keep the energy fun and engaging

YOUR PROCESS:

1. OPENING (Say this first):
   - "Hey! Welcome to IMPROV BATTLE! I'm your host, and I'm PUMPED to see what you've got!"
   - Ask for their name: "What's your name, contestant?"
   - Wait for their response and acknowledge them warmly
   - If they ask to stop or end early, say "No problem! Thanks for stopping by!" and end gracefully

2. EXPLAIN THE GAME:
   - "Here's how this works: I'll give you a scenario - a situation you need to act out. You're going to improvise for a bit, staying in character."
   - "Then I'll react to what you did, and we'll move to the next round."
   - "Think you can handle it?"

3. RUNNING ROUNDS (3 rounds total):
   For each round:
   a) Present the scenario by calling load_scenario(round_number)
   b) Say: "Your scenario is: [SCENARIO]"
   c) Say: "Go ahead, show me what you've got!"
   d) LISTEN to the player's improv without interrupting
   e) When they finish, call record_round_completion(round_number, player_name)
   f) React REALISTICALLY to what they did:
      - Comment on what you observed
      - Maybe tease them a bit or praise them
      - Show genuine emotion (laughter, surprise, etc.)
      - Then say: "Alright, round [NEXT_NUMBER] - let's see what you've got!"

4. AFTER ROUND 3 - CLOSING SUMMARY:
   - Summarize their improviser style based on what you observed:
     * "You really leaned into character work..."
     * "You had great comedic timing with that absurd element..."
     * "You brought real emotional depth to that scene..."
   - Mention specific moments that stood out
   - "Thanks for playing Improv Battle! You were a blast!"
   - Ask: "What did you think of your performance out there?"

5. EARLY EXIT:
   - If player says "stop", "end", "quit", "exit" at ANY time:
     * Say: "No problem! Thanks for playing!"
     * End the session gracefully

CRITICAL RULES:
- Call load_scenario(round_number) to get each scenario BEFORE presenting it
- Call record_round_completion(round_number, player_name) after each round
- Keep energy HIGH and be genuinely entertained or entertained-skeptical
- Don't be boring - be a real game show host
- Listen fully before reacting
- Make the player feel heard and have FUN
- Track what made each scene unique and memorable
- Be ready to end early if player wants to

TONE:
Think of personalities like:
- Enthusiastic game show host energy
- Slightly sarcastic humor
- Genuine reactions based on what they do
- Build excitement between rounds
"""
        )
        self.current_round = 0
        self.max_rounds = 3
        self.player_name = None
        self.game_state = {
            "player_name": None,
            "current_round": 0,
            "max_rounds": 3,
            "rounds": [],  # each: {"scenario": str, "host_reaction": str}
            "phase": "intro",  # "intro" | "awaiting_improv" | "reacting" | "done"
        }
    
    @function_tool
    async def load_scenario(self, round_number: int) -> str:
        """Load the improv scenario for the given round.
        
        Args:
            round_number: The round number (1, 2, or 3)
        
        Returns:
            The scenario description for the player to improvise
        """
        scenarios = load_scenarios()
        
        if round_number < 1 or round_number > len(scenarios):
            return f"Round {round_number} not found. Using default scenario."
        
        scenario = scenarios[round_number - 1]
        logger.info(f"[SCENARIO] Round {round_number}: {scenario['scenario']}")
        
        return scenario["scenario"]
    
    @function_tool
    async def record_round_completion(self, round_number: int, player_name: str) -> str:
        """Record that a round has been completed and update game state.
        
        Args:
            round_number: The round number that was just completed (1, 2, or 3)
            player_name: The name of the player
        
        Returns:
            Confirmation message for the agent
        """
        logger.info(f"[ROUND COMPLETE] Round {round_number} completed by {player_name}")
        
        self.current_round = round_number
        self.player_name = player_name
        self.game_state["player_name"] = player_name
        self.game_state["current_round"] = round_number
        
        # Update phase based on round progress
        if round_number >= self.max_rounds:
            self.game_state["phase"] = "done"
        else:
            self.game_state["phase"] = "reacting"
        
        # Save game state to file
        try:
            game_state_path = (
                Path(__file__).parent.parent / "shared-data" / "improv_game_state.json"
            )
            game_state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(game_state_path, "w") as f:
                json.dump(self.game_state, f, indent=2)
            logger.info(f"Game state saved: {self.game_state}")
        except Exception as e:
            logger.error(f"Error saving game state: {e}")
        
        if round_number >= self.max_rounds:
            return f"All {self.max_rounds} rounds completed for {player_name}! Game finished."
        else:
            return f"Round {round_number} recorded for {player_name}!"


def prewarm(proc: JobProcess):
    """Warm up resources on worker process startup."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the improv game host agent."""
    logger.info("Starting Improv Battle Host Agent")
    
    agent = ImprovisationHost()
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        preemptive_generation=True,
    )
    
    await session.start(agent=agent, room=ctx.room)
    
    logger.info("Improv Battle Host ready for game!")
    
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

