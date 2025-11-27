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


class FraudAlertAgent(Agent):
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
    - IMPORTANT: Call the 'verify_customer' function after asking the security question
    - Say: "To verify your identity, I have a security question for you: [securityQuestion]"
    - Wait for their answer
    - Call: verify_customer(answer) with their response
    - If verification fails (function returns false):
        - Say: "I apologize, but I'm unable to verify your identity at this time. For security reasons, please call our main line to confirm this transaction. Thank you for banking with us."
        - End the conversation

    4. READ SUSPICIOUS TRANSACTION (Only if verification passes):
    - Say: "Thank you for verifying. Here's the transaction we're investigating:"
    - Read out clearly: "We detected a [transactionCategory] transaction for [transactionAmount] at [transactionName] on [transactionTime] from [transactionSource]. This transaction was made in [location]."
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
        
       
        self.company_name = "Razorpay"
        self.company_info = {
            "about": "Razorpay is India's leading payment technology company founded in 2014. We help businesses of all sizes accept payments online, in-app, and in-store with our powerful payment solutions.",
            "what_we_do": "We provide payment gateway solutions that let businesses accept credit cards, debit cards, net banking, wallets, and more. We also offer payouts, invoicing, and subscription management.",
            "who_we_serve": "E-commerce stores, SaaS platforms, marketplaces, education platforms, healthcare providers, and any online business that needs to accept payments.",
            "pricing_model": "We charge a transaction fee (typically 2-3% depending on payment method) with no setup fees. No monthly charges. Pay only when you process payments.",
        }
        
        self.faq = [
            {
                "question": "What does Razorpay do?",
                "answer": "Razorpay is a payment gateway that helps businesses accept online payments. We support credit cards, debit cards, net banking, UPI, wallets, and more. Whether you're an e-commerce store, app, or marketplace, we make it easy to accept payments from customers."
            },
            {
                "question": "Who is Razorpay for?",
                "answer": "Razorpay is for any business that needs to accept online payments. This includes e-commerce stores, SaaS platforms, marketplaces, education platforms, healthcare providers, subscription businesses, and more. We serve startups to large enterprises."
            },
            {
                "question": "How much does Razorpay cost?",
                "answer": "There are no setup fees or monthly charges. You only pay a transaction fee, typically 2-3% depending on the payment method (credit card, debit card, UPI, etc.). It's a pay-as-you-grow model."
            },
            {
                "question": "Do you offer a free trial?",
                "answer": "Yes! You can start with a free test account to integrate and test our payment gateway. Once you go live, you pay only for actual transactions you process."
            },
            {
                "question": "How long does integration take?",
                "answer": "Integration typically takes 30 minutes to a few hours depending on your technical setup. We have plugins for popular platforms like WooCommerce, Shopify, and custom integration guides for others."
            },
            {
                "question": "Is Razorpay secure?",
                "answer": "Yes, Razorpay is PCI-DSS Level 1 certified, the highest security standard for payment processing. We encrypt all sensitive data and comply with international security standards."
            },
            {
                "question": "What payment methods do you support?",
                "answer": "We support credit cards, debit cards, net banking, UPI, digital wallets (Apple Pay, Google Pay), and more. This gives your customers multiple payment options."
            },
            {
                "question": "Do you offer international payments?",
                "answer": "Yes, we support payments from international cards and customers. We also provide international payouts through our Razorpay X product."
            },
            {
                "question": "What about payouts and invoicing?",
                "answer": "Beyond payments, Razorpay X offers payouts (send money to vendors/partners), invoicing, automated billing, and business banking. It's a complete payments platform."
            },
            {
                "question": "How is customer support?",
                "answer": "We offer 24/7 customer support via chat, email, and phone. Our support team is highly responsive and helps with integration, troubleshooting, and any questions."
            }
        ]
        
        self.lead_data = {
            "name": None,
            "company": None,
            "email": None,
            "role": None,
            "use_case": None,
            "team_size": None,
            "timeline": None,
        }
        
        logger.info(f"Initialized {self.company_name} SDR with {len(self.faq)} FAQ entries")
    
    @function_tool
    async def search_faq(self, question: str):
        """Search the FAQ for relevant answers about Razorpay.
        
        Call this function when the user asks about our product, pricing, features, integration, security, etc.
        This searches the FAQ and returns the most relevant answer based on the user's question.
        
        Args:
            question: The user's question about Razorpay (e.g., "What do you do?", "How much does it cost?")
        
        Returns:
            The relevant FAQ answer or a helpful message if the answer isn't in our FAQ.
        """
        
        if not question:
            return "I didn't catch your question. Could you please ask me about Razorpay?"
        
        question_lower = question.lower()
        best_match = None
        best_score = 0
        
        # Search through FAQ entries
        for faq_entry in self.faq:
            faq_question = faq_entry["question"].lower()
            faq_answer = faq_entry["answer"].lower()
            
            # Count keyword matches
            score = 0
            for word in question_lower.split():
                if len(word) > 2:
                    if word in faq_question:
                        score += 2  # Question match is worth more
                    if word in faq_answer:
                        score += 1  # Answer match
            
            if score > best_score:
                best_score = score
                best_match = faq_entry
        
        if best_match and best_score > 0:
            logger.info(f"Found FAQ answer for: {best_match['question']} (score: {best_score})")
            return f"Great question! {best_match['answer']}"
        else:
            # If no match, return a general response
            logger.info(f"No direct FAQ match for: {question}")
            return f"That's a great question about Razorpay! I want to give you accurate information. Feel free to ask about our payment solutions, pricing, integration, security, or how we can help your business."
    
    @function_tool
    async def collect_lead_info(self, field_name: str, field_value: str):
        """Collect and store a lead field during the conversation.
        
        Call this function to store lead information as the user provides it naturally.
        This saves: name, company, email, role, use_case, team_size, timeline
        
        Args:
            field_name: The type of info (name, company, email, role, use_case, team_size, or timeline)
            field_value: The user's response for that field
        
        Returns:
            Confirmation that the info was saved.
        """
        
        field_name = field_name.lower()
        
        if field_name in self.lead_data:
            self.lead_data[field_name] = field_value
            logger.info(f"Stored lead info - {field_name}: {field_value}")
            return f"Got it! I've saved that {field_name}."
        else:
            return f"I'm not sure what field '{field_name}' is. I collect: name, company, email, role, use_case, team_size, and timeline."
    
    @function_tool
    async def finalize_lead(self):
        """Generate end-of-call summary and save the lead to JSON file.
        
        Call this function when the user indicates they're done with the call.
        This creates a verbal summary, saves the lead data to lead_captures.json, and concludes the call.
        
        Returns:
            A warm closing message with a summary of what was discussed.
        """
        
        summary_parts = []
        
        if self.lead_data["name"]:
            summary_parts.append(f"your name is {self.lead_data['name']}")
        if self.lead_data["company"]:
            summary_parts.append(f"you work at {self.lead_data['company']}")
        if self.lead_data["role"]:
            summary_parts.append(f"you're a {self.lead_data['role']}")
        if self.lead_data["use_case"]:
            summary_parts.append(f"you want to use Razorpay for {self.lead_data['use_case']}")
        if self.lead_data["team_size"]:
            summary_parts.append(f"your team size is {self.lead_data['team_size']}")
        if self.lead_data["timeline"]:
            summary_parts.append(f"your timeline is {self.lead_data['timeline']}")
        
        if summary_parts:
            summary_text = "So to recap: " + ", ".join(summary_parts) + ". We'll follow up with you soon with more details about how Razorpay can help!"
        else:
            summary_text = "Thanks so much for chatting with me! We'll follow up soon with information about Razorpay. Looking forward to working with you!"
        
        self._save_lead_to_json()
        
        logger.info(f"Lead finalized and saved: {self.lead_data}")
        
        return summary_text
    
    def _save_lead_to_json(self):
        """Save collected lead data to lead_captures.json file."""
        
        lead_record = {
            "timestamp": datetime.now().isoformat(),
            "company_name": self.company_name,
            "name": self.lead_data["name"],
            "email": self.lead_data["email"],
            "role": self.lead_data["role"],
            "use_case": self.lead_data["use_case"],
            "team_size": self.lead_data["team_size"],
            "timeline": self.lead_data["timeline"],
            "prospect_company": self.lead_data["company"],
        }
        
        leads_file = Path("shared-data/lead_captures.json")
        leads_file.parent.mkdir(parents=True, exist_ok=True)
        
        if leads_file.exists():
            try:
                with open(leads_file, 'r') as f:
                    leads = json.load(f)
            except:
                leads = []
        else:
            leads = []
        
        leads.append(lead_record)
        
        with open(leads_file, 'w') as f:
            json.dump(leads, f, indent=2)
        
        logger.info(f"Lead saved to {leads_file}")


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
        agent=FraudAlertAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
