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
            instructions="""You are a professional, calm, and reassuring Fraud Detection Representative for SecureBank.

                YOUR ROLE:
                - You work in the bank's fraud prevention department
                - Your job is to verify suspicious transactions with customers
                - Be empathetic, professional, and never accusatory
                - Use a warm but formal tone

                YOUR PROCESS:

                1. GREETING (Start immediately when call begins):
                - Greet warmly: "Hello, this is SecureBank Fraud Prevention Department. We've detected a suspicious transaction on your account and need to verify it with you. May I have your name please?"
                - Listen for their name

                2. LOAD FRAUD CASE:
                - Call the 'load_fraud_case' function with the name they provided
                - This retrieves the suspicious transaction details from our system
                - If no case found, say: "I'm sorry, I couldn't find an account with that name. Please call our main line at [number]. Thank you."

                3. VERIFY IDENTITY (Security Question):
                - IMPORTANT: First call 'get_security_question' to retrieve the actual security question
                - Say: "To verify your identity, I have a security question for you: " and then ask the question returned by the function
                - Wait for their answer
                - Call: verify_customer(answer) with their response
                - If verification fails (function returns false):
                    - Say: "I apologize, but I'm unable to verify your identity at this time. For security reasons, please call our main line to confirm this transaction. Thank you for banking with us."
                    - End the conversation

                4. READ SUSPICIOUS TRANSACTION (Only if verification passes):
                - Say: "Thank you for verifying. Here's the transaction we're investigating:"
                - Call: get_transaction_details() to retrieve the formatted transaction details from the database
                - Read the transaction details exactly as returned by the function
                - Say: "Did you authorize this transaction? Please say yes or no."

                5. HANDLE THEIR RESPONSE:
                - If they say YES (they made it):
                    - Say: "Thank you for confirming. This transaction is marked as legitimate. Your account is secure."
                    - Call: confirm_transaction(true)
                
                - If they say NO (they didn't make it):
                    - Say: "Thank you for reporting this. We're immediately blocking your card and initiating a fraud dispute. You should receive a replacement card within 3-5 business days."
                    - Call: confirm_transaction(false)

                6. CLOSE THE CALL:
                - Wait for the function to return the confirmation message
                - Say: "We've updated your account. Thank you for being vigilant about your security. Is there anything else I can help you with?"
                - If they say no, say: "Thank you for banking with us. Goodbye."

                CRITICAL RULES:
                - NEVER ask for full card numbers, PIN, password, CVV, or any sensitive information
                - ONLY use the database fields for verification (security question only)
                - Be professional and calm - never accusatory
                - Always call the appropriate function at the right time
                - If the function fails, explain in a reassuring way
                - Use the exact transaction details from the database, don't improvise
                - Keep the conversation conversational but professional
                - Be empathetic - fraud is stressful for customers
            """
            )
        
        self.current_fraud_case = None
        
    @function_tool
    async def load_fraud_case(self, userName: str = None):
        """Load a fraud case from the database by customer name.
    
            This function retrieves a suspicious transaction case from the fraud database
            based on the customer's name. It's called after the customer provides their name
            during the fraud verification call.
            
            Args:
                userName: The customer's full name (e.g., "John Doe")
            
            Returns:
                A dictionary containing the fraud case details including:
                - transactionAmount: The amount of the suspicious transaction
                - transactionName: The merchant name (e.g., "Amazon Purchase")
                - transactionTime: When the transaction occurred
                - transactionSource: Where the transaction originated from
                - location: Geographic location of the transaction
                - securityQuestion: A security question for identity verification
                - cardEnding: Last 4 digits of the card (masked)
                
                If no case is found, returns an error message asking them to call the bank.
        """
        
        fraud_cases = Path(__file__).parent.parent / "shared-data" / "fraud_cases.json"
        if fraud_cases.exists():
            try:
                with open (fraud_cases, 'r') as f:
                    frauds = json.load(f)
                    
            except:
                frauds=[]
        else:
            frauds=[]
            
        for fraud in frauds:
            if fraud['userName'].lower() == userName.lower():
                self.current_fraud_case = fraud
        
        if self.current_fraud_case:
            logger.info(f"Loaded fraud case for: {userName}")
            return "Great! I found your account. Let me verify your identity first."
    
        logger.warning(f"No fraud case found for: {userName}")
        return "I'm sorry, I couldn't find an account with that name. Please call our main line. Thank you."
            
    @function_tool
    async def get_security_question(self):
        """Get the security question for the loaded fraud case.
        
        This function retrieves the security question from the currently loaded
        fraud case and returns it to ask the customer.
        
        Returns:
            The security question string (e.g., "What is your mother's maiden name?")
        """
        if self.current_fraud_case:
            question = self.current_fraud_case['securityQuestion']
            logger.info(f"Retrieved security question for case {self.current_fraud_case['id']}")
            return question
        else:
            return "Unable to retrieve security question. Please try again."
        
        
    @function_tool
    async def verify_customer(self, answer: str = None):
        """Verify the customer's identity by checking their security question answer.
    
            This function compares the customer's answer to the security question with
            the correct answer stored in the fraud case database. It's used to confirm
            the customer is who they claim to be before revealing fraud details.
            
            Args:
                answer: The customer's answer to the security question (e.g., "sharma")
            
            Returns:
                A boolean:
                - True: If the answer matches the stored answer (identity verified)
                - False: If the answer is incorrect (verification failed)
                
                When False, the call should be ended for security reasons and the 
                customer should be directed to call the bank's main line.
        """
        
        correct_answer = self.current_fraud_case['securityAnswer']
        if answer.lower() == correct_answer.lower():
            verification_passed = True
        else:
            verification_passed = False
            
        if verification_passed:
            logger.info(f"Customer verified successfully")
            return True
        
        else:
            logger.info("Verification failed for customer")
            return False

    @function_tool
    async def get_transaction_details(self):
        """Get the formatted transaction details for the loaded fraud case.
        
        This function retrieves and formats the suspicious transaction details
        from the loaded fraud case so the agent can read them exactly as stored
        in the database.
        
        Returns:
            A formatted string with the transaction details including amount, 
            merchant name, time, category, source, and location.
        """
        if self.current_fraud_case:
            details = f"We detected a {self.current_fraud_case['transactionCategory']} transaction for {self.current_fraud_case['transactionAmount']} at {self.current_fraud_case['transactionName']} on {self.current_fraud_case['transactionTime']} from {self.current_fraud_case['transactionSource']}. This transaction was made in {self.current_fraud_case['location']}."
            logger.info(f"Retrieved transaction details for case {self.current_fraud_case['id']}")
            return details
        else:
            return "Unable to retrieve transaction details. Please try again."
        
    @function_tool
    async def confirm_transaction(self, is_legitimate: bool = None):
        """Record the customer's confirmation about the suspicious transaction and update the database.
    
            This function takes the customer's yes/no response about whether they made the
            suspicious transaction, updates the fraud case status in the database, and
            returns an appropriate confirmation message.
            
            Args:
                is_legitimate: 
                    - True: Customer confirms they made the transaction (it's legitimate)
                    - False: Customer denies making the transaction (it's fraudulent)
            
            Returns:
                A confirmation message including:
                - What action was taken (transaction marked safe OR card blocked & dispute initiated)
                - Timeline (if applicable, e.g., "replacement card in 3-5 business days")
                
                The function also updates the fraud case in the database with:
                - status: Changed from "pending_review" to "confirmed_safe" or "confirmed_fraud"
                - outcome: The resolution (safe/fraudulent/verification_failed)
                - outcomeNote: A detailed note about what action was taken
        """
        
        if is_legitimate:
            self.current_fraud_case["status"] = "confirmed_safe"
            self.current_fraud_case["outcome"] = "safe"
            self.current_fraud_case["outcomeNote"] = "Customer confirmed transaction as legitimate."
            
        else:
            self.current_fraud_case["status"] = "confirmed_fraud"
            self.current_fraud_case["outcome"] = "fraudulent"
            self.current_fraud_case["outcomeNote"] = "Customer denied transaction. Card blocked and dispute initiated."
            
        fraud_cases = Path(__file__).parent.parent / "shared-data" / "fraud_cases.json"
        with open (fraud_cases, 'r') as f:
            frauds = json.load(f)
            
        for fraud in frauds:
            if fraud['id'] == self.current_fraud_case['id']:
                fraud.update(self.current_fraud_case)  
                
        with open(fraud_cases, 'w') as f:
            json.dump(frauds, f, indent=2)
                
        logger.info(f"Transaction marked as {self.current_fraud_case['outcome']} for case {self.current_fraud_case['id']}")
                
        if is_legitimate:
            return "Transaction confirmed as safe. Your account is secure. Thank you."
        
        else:
            return "We've blocked your card and initiated a dispute. You'll receive a replacement card in 3-5 business days."

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
