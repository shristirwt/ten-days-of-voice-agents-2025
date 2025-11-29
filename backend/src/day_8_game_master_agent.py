import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import uuid

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


class GameMasterAgent(Agent):
    def __init__(self, room=None) -> None:
        self.world_state = {
            "universe": "Eldoria - A land of magic, dragons, and ancient ruins",
            "current_location": "The Wandering Wyvern tavern, Seahaven port city",
            "characters": {
                "player": {
                    "name": "Unknown adventurer",
                    "class": "Unknown",
                    "hp": 20,
                    "inventory": [],
                    "traits": []
                },
                "npcs": {}
            },
            "events": [],
            "quests": [],
            "turn_count": 0
        }
        self.room = room
        
        super().__init__(
            instructions="""You are an immersive and dramatic Game Master running a fantasy D&D-style adventure. Your role is to guide the player through an epic story set in the mystical realm of Eldoria.

                WORLD SETTING:
                - You are the Game Master of Eldoria, a land of magic, ancient ruins, and dangerous creatures
                - The story takes place in a fantasy medieval world with magic, dragons, and adventure
                - Be vivid, dramatic, and immersive in your descriptions
                - Use fantasy language and create atmosphere

                YOUR TOOLS:
                - Call get_world_state() at the start of each turn to review what's happened so far
                - Call set_player_name(name) when the player tells you their character's name
                - Call set_current_location(location, description) when the player moves to a new area
                - Call update_world_state(field, key, value) to record NPCs, events, and quests
                - Example: update_world_state("characters", "Thorne the Tavern Keeper", "A scarred innkeeper with a mysterious past")
                - Example: update_world_state("events", "", "Player met Thorne and learned of the missing artifact")

                YOUR ROLE AND PROCESS:

                1. OPENING SCENE (First message):
                - Greet the player warmly: "Welcome, adventurer! I am your Game Master, and I will guide you through the realm of Eldoria."
                - Set the scene: Describe the starting location vividly
                - Example: "You find yourself in the bustling tavern called 'The Wandering Wyvern' in the port city of Seahaven. The smell of ale and roasted meat fills the air. Around you, merchants, sailors, and adventurers chat and laugh. A cloaked figure in the corner catches your eye..."
                - End with: "What do you do?"

                2. ACTIVE STORYTELLING:
                - Listen to what the player wants to do
                - Describe the consequences of their actions in vivid detail
                - Create atmosphere with sensory descriptions (sights, sounds, smells, feelings)
                - Example: "As you approach the cloaked figure, the tavern noise fades. You notice the mysterious stranger's hand rests on a concealed dagger..."
                - Always respond to player choices meaningfully

                3. CREATE TENSION AND CHALLENGE:
                - Introduce challenges and interesting encounters
                - NPCs with distinct personalities (the suspicious tavern keeper, the mysterious mage, etc.)
                - Environmental hazards and puzzles
                - Opportunities for the player to make meaningful choices

                4. DIALOGUE AND NPC INTERACTION:
                - When NPCs speak, use distinct voices and personalities in your descriptions
                - Example: "The tavern keeper, a burly man with a scarred face, leans over the bar. 'You look like trouble, friend. That's good. I've got a job that needs someone like you.'"
                - Always describe NPC reactions to player choices

                5. PROGRESSION OF THE STORY:
                - Build toward a mini-arc or mini-quest
                - After 3-4 exchanges, introduce a main objective or challenge
                - Examples: Find a lost artifact, help an NPC, escape danger, solve a mystery
                - Create natural pacing with calm moments and action sequences

                6. STORY CONSISTENCY AND WORLD STATE:
                - Use the world state to track important story elements
                - Call update_world_state() to record NPCs, events, and quests as they develop
                - Remember what has happened in previous turns
                - Maintain consistency (if an NPC is dead, they stay dead; if an item was taken, it's gone)
                - The world should feel alive and reactive to player choices

                7. SESSION ENDING:
                - After 8-15 exchanges, guide the story toward a conclusion
                - Reach a natural conclusion (success, failure, or cliffhanger)
                - Summarize what happened in this session
                - Offer the chance to continue or create a new adventure

                CRITICAL RULES:
                - NEVER break character as Game Master
                - ALWAYS end messages with "What do you do?" or similar prompt
                - Use vivid, sensory descriptions
                - Be creative and respond to player choices meaningfully
                - Keep track of story elements (names, locations, events)
                - Balance challenge with fun
                - Be dramatic and engaging - this is entertainment!
                - Create a sense of progression and advancement
            """
        )
    
    @function_tool
    async def update_world_state(
        self,
        field: str,
        key: str,
        value: str
    ) -> str:
        """Update the game world state (characters, locations, events, quests)
        
        Args:
            field: 'characters', 'locations', 'events', or 'quests'
            key: the identifier (e.g., character name or location name)
            value: the data to store
        
        Returns:
            Confirmation message
        """
        try:
            if field == "characters":
                if "npcs" not in self.world_state["characters"]:
                    self.world_state["characters"]["npcs"] = {}
                self.world_state["characters"]["npcs"][key] = value
                logger.info(f"✓ Added NPC: {key}")
                return f"Added character '{key}' to the world."
            
            elif field == "locations":
                if "locations" not in self.world_state:
                    self.world_state["locations"] = {}
                self.world_state["locations"][key] = value
                logger.info(f"✓ Added location: {key}")
                return f"Noted location: '{key}'."
            
            elif field == "events":
                self.world_state["events"].append({"turn": self.world_state["turn_count"], "event": value})
                logger.info(f"✓ Event recorded: {value}")
                return f"Event recorded: {value}"
            
            elif field == "quests":
                self.world_state["quests"].append({"quest": key, "status": value})
                logger.info(f"✓ Quest added: {key}")
                return f"Quest added: '{key}' - {value}"
            
            else:
                return "Invalid field. Use: characters, locations, events, or quests"
        except Exception as e:
            logger.error(f"Error updating world state: {e}")
            return f"Error: {str(e)}"
    
    @function_tool
    async def get_world_state(self) -> str:
        """Retrieve current game world state for story consistency
        
        Returns:
            Formatted world state summary
        """
        self.world_state["turn_count"] += 1
        
        summary = f"""📖 WORLD STATE (Turn {self.world_state['turn_count']}):
🌍 Universe: {self.world_state['universe']}
📍 Current Location: {self.world_state['current_location']}

👥 Characters:
- Player: {self.world_state['characters']['player']['name']} (Class: {self.world_state['characters']['player']['class']}, HP: {self.world_state['characters']['player']['hp']})"""
        
        if self.world_state['characters']['npcs']:
            summary += "\n- NPCs: " + ", ".join(self.world_state['characters']['npcs'].keys())
        
        if self.world_state['events']:
            summary += f"\n\n📜 Recent Events ({len(self.world_state['events'])} total):"
            for event in self.world_state['events'][-3:]:
                summary += f"\n  • {event['event']}"
        
        if self.world_state['quests']:
            summary += f"\n\n⚔️ Active Quests:"
            for quest in self.world_state['quests']:
                summary += f"\n  • {quest['quest']}: {quest['status']}"
        
        logger.info(f"World state retrieved for turn {self.world_state['turn_count']}")
        return summary
    
    @function_tool
    async def set_player_name(self, name: str) -> str:
        """Set or update the player character's name
        
        Args:
            name: Character name
        
        Returns:
            Confirmation message
        """
        self.world_state['characters']['player']['name'] = name
        logger.info(f"✓ Player name set to: {name}")
        return f"Adventurer '{name}' has entered the realm of Eldoria!"
    
    @function_tool
    async def set_current_location(self, location_name: str, description: str) -> str:
        """Update the player's current location in the story
        
        Args:
            location_name: Name of the location
            description: Brief description of the location
        
        Returns:
            Confirmation message
        """
        self.world_state['current_location'] = location_name
        if "locations" not in self.world_state:
            self.world_state["locations"] = {}
        self.world_state["locations"][location_name] = description
        logger.info(f"✓ Location changed to: {location_name}")
        return f"You are now at: {location_name}. {description}"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Narration",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Create agent instance
    agent = GameMasterAgent(room=ctx.room)

    # Start the session with the Game Master agent
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
