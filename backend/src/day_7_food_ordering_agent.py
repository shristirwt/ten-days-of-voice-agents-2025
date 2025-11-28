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


class FoodOrderingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly and helpful Food & Grocery Ordering Assistant for FreshCart, your favorite online food and grocery delivery service.

                YOUR ROLE:
                - Help customers order groceries, snacks, and prepared foods
                - Maintain a shopping cart throughout the conversation
                - Provide recommendations and answer questions about items
                - Process orders when customers are ready

                YOUR PROCESS:

                1. GREETING:
                - Greet warmly: "Hello! Welcome to FreshCart. I'm your shopping assistant. What can I help you order today? We have groceries, snacks, prepared foods, and drinks."
                - Listen for what they want to order

                2. UNDERSTAND THEIR REQUEST:
                - If they ask for specific items: "I'll add that to your cart."
                - If they ask for "ingredients for [dish]": Intelligently add multiple related items
                  - Example: "ingredients for pasta" → add pasta, sauce, and cheese
                  - Example: "ingredients for a sandwich" → add bread and deli meat
                - If they ask about items: "Let me search our catalog for you."
                - Always confirm what you're adding: "I've added 2 loaves of bread to your cart."

                3. MANAGE THE CART:
                - Keep track of items, quantities, and prices
                - When they ask "What's in my cart?" or "Show me my cart":
                  - Call: get_cart_summary() to get formatted cart details
                  - Read the summary exactly as returned
                - Support cart operations:
                  - Adding items: "I've added [item] to your cart."
                  - Removing items: "I've removed [item] from your cart."
                  - Updating quantities: "I've updated [item] to [quantity]."

                4. HANDLE SPECIAL REQUESTS:
                - Dietary preferences: "I can filter for vegan, gluten-free, or organic items."
                - Budget concerns: "I'll help you find items within your budget."
                - Allergies/preferences: "I'll keep that in mind and won't suggest [item]."

                5. DETECT ORDER COMPLETION:
                - Listen for phrases like:
                  - "That's all"
                  - "I'm done"
                  - "Place my order"
                  - "Check out"
                  - "Complete the order"
                - When you detect this, say: "Great! Let me confirm your order."

                6. CONFIRM AND PLACE ORDER:
                - Call: get_cart_summary() to show final cart
                - Call: place_order() to save the order to JSON
                - Announce: "Your order has been placed! You can expect delivery in 30-45 minutes."
                - Provide confirmation: "Thank you for ordering with FreshCart!"

                7. POST-ORDER:
                - If they ask "Where's my order?": Offer to check status
                - If they want to add more: "Would you like to start a new order or add more items?"
                - Be helpful and friendly throughout

                CRITICAL RULES:
                - ALWAYS confirm items before adding to cart
                - ALWAYS use function tools to manage cart - never make up cart contents
                - Keep quantities reasonable (1-10 items per product typically)
                - Be conversational but professional
                - If unsure about an item, ask clarifying questions
                - Never add items without explicit or clear implied consent
                - Use exact prices and item names from the catalog
                - Be empathetic about delivery times and policies
            """
        )
        self.cart = {}  
        
    @function_tool
    async def search_catalog(self, query: str = None):
        """Search for items in the catalog by name, category, or tags.
        
        This function searches the food catalog for items matching the query.
        Can search by item name, category, or tags.
        
        Args:
            query: Search term (e.g., "bread", "pasta", "vegan", "breakfast")
        
        Returns:
            A formatted list of matching items with names, prices, categories, and tags
        """
        catalog_path = Path(__file__).parent.parent / "shared-data" / "catalog.json"
        
        try:
            with open(catalog_path, 'r') as f:
                catalog = json.load(f)
        except:
            return "Sorry, I couldn't access the catalog. Please try again."
        
        query_lower = query.lower()
        results = []
        
        for item in catalog:
            name_match = query_lower in item['name'].lower()
            category_match = query_lower in item['category'].lower()
            tags_match = any(query_lower in tag.lower() for tag in item.get('tags', []))
            
            if name_match or category_match or tags_match:
                results.append(item)
        
        if not results:
            logger.warning(f"No items found for query: {query}")
            return f"I couldn't find any items matching '{query}'. Would you like to try a different search?"
        
        response = f"I found {len(results)} items for you:\n"
        for item in results[:10]:  # Limit to 10 results
            response += f"- {item['name']}: ₹{item['price']} ({item['category']})\n"
        
        logger.info(f"Search results for '{query}': {len(results)} items found")
        return response

    @function_tool
    async def add_to_cart(self, item_name: str = None, quantity: int = 1):
        """Add an item to the cart with specified quantity.
        
        This function adds an item from the catalog to the shopping cart.
        Searches for the item by name and adds the specified quantity.
        
        Args:
            item_name: Name of the item to add (e.g., "Whole Wheat Bread")
            quantity: How many to add (default 1)
        
        Returns:
            Confirmation message with item name, quantity, and price
        """
        catalog_path = Path(__file__).parent.parent / "shared-data" / "catalog.json"
        
        try:
            with open(catalog_path, 'r') as f:
                catalog = json.load(f)
        except:
            return "Sorry, I couldn't access the catalog. Please try again."
        
        item_found = None
        for item in catalog:
            if item['name'].lower() == item_name.lower():
                item_found = item
                break
        
        if not item_found:
            logger.warning(f"Item not found: {item_name}")
            return f"I couldn't find '{item_name}' in our catalog. Would you like me to search for something similar?"
        
        item_id = item_found['id']
        if item_id in self.cart:
            self.cart[item_id]['quantity'] += quantity
        else:
            self.cart[item_id] = {
                'name': item_found['name'],
                'quantity': quantity,
                'price': item_found['price'],
                'category': item_found['category']
            }
        
        total_price = self.cart[item_id]['quantity'] * self.cart[item_id]['price']
        logger.info(f"Added {quantity}x {item_name} to cart. Total: ₹{total_price}")
        return f"Great! I've added {quantity} {item_found['name']} to your cart at ₹{item_found['price']} each."

    @function_tool
    async def remove_from_cart(self, item_name: str = None):
        """Remove an item completely from the cart.
        
        This function removes all quantities of an item from the shopping cart.
        
        Args:
            item_name: Name of the item to remove (e.g., "Whole Wheat Bread")
        
        Returns:
            Confirmation message about the removal
        """
        item_id = None
        for id, item in self.cart.items():
            if item['name'].lower() == item_name.lower():
                item_id = id
                break
        
        if not item_id:
            logger.warning(f"Item not in cart: {item_name}")
            return f"I don't see '{item_name}' in your cart. Would you like to add something else?"
        
        removed_item = self.cart.pop(item_id)
        logger.info(f"Removed {removed_item['name']} from cart")
        return f"Done! I've removed {removed_item['name']} from your cart."

    @function_tool
    async def update_quantity(self, item_name: str = None, quantity: int = 1):
        """Update the quantity of an item in the cart.
        
        This function changes the quantity of an existing cart item.
        
        Args:
            item_name: Name of the item to update
            quantity: New quantity (must be positive)
        
        Returns:
            Confirmation message with new quantity and total
        """
        if quantity < 1:
            return "Quantity must be at least 1. If you want to remove this item, just say so!"
        
        item_id = None
        for id, item in self.cart.items():
            if item['name'].lower() == item_name.lower():
                item_id = id
                break
        
        if not item_id:
            logger.warning(f"Item not in cart: {item_name}")
            return f"I don't see '{item_name}' in your cart. Would you like to add it first?"
        
        old_quantity = self.cart[item_id]['quantity']
        self.cart[item_id]['quantity'] = quantity
        total_price = quantity * self.cart[item_id]['price']
        
        logger.info(f"Updated {item_name} quantity from {old_quantity} to {quantity}")
        return f"Updated! You now have {quantity} {item_name} in your cart at a total of ₹{total_price}."

    @function_tool
    async def get_cart_summary(self):
        """Get a formatted summary of the current shopping cart.
        
        This function returns a complete summary of all items in the cart,
        their quantities, prices, and the total amount.
        
        Returns:
            A formatted string with all cart items and the total price
        """
        if not self.cart:
            return "Your cart is empty. What would you like to order?"
        
        summary = "Here's what's in your cart:\n"
        grand_total = 0
        
        for item_id, item in self.cart.items():
            item_total = item['quantity'] * item['price']
            grand_total += item_total
            summary += f"- {item['quantity']}x {item['name']}: ₹{item_total}\n"
        
        summary += f"\nTotal: ₹{grand_total}"
        logger.info(f"Cart summary requested. Items: {len(self.cart)}, Total: ₹{grand_total}")
        return summary

    @function_tool
    async def get_recipe_ingredients(self, dish: str = None):
        """Get a list of ingredients needed for a specific dish.
        
        This function returns the items needed to prepare a dish.
        Used when customers ask for "ingredients for [dish]".
        
        Args:
            dish: Name of the dish (e.g., "pasta", "sandwich", "breakfast")
        
        Returns:
            A list of items to add to the cart for the dish
        """
        catalog_path = Path(__file__).parent.parent / "shared-data" / "catalog.json"
        
        try:
            with open(catalog_path, 'r') as f:
                catalog = json.load(f)
        except:
            return "Sorry, I couldn't access the catalog. Please try again."
        
        recipes = {
            "pasta": ["Spaghetti Pasta (500g)", "Tomato Pasta Sauce (400ml)", "Cheddar Cheese (200g)"],
            "sandwich": ["Whole Wheat Bread", "Chicken Deli Meat (200g)", "Cheddar Cheese (200g)"],
            "peanut butter sandwich": ["Whole Wheat Bread", "Peanut Butter (500g)"],
            "breakfast": ["Eggs (12 pack)", "Whole Wheat Bread", "Fresh Milk (1L)", "Granola Cereal (400g)"],
            "pizza": ["Frozen Margherita Pizza"],
            "salad": ["Fresh Spinach (500g)", "Fresh Tomatoes (1kg)", "Red Apples (1kg)"],
            "coffee": ["Instant Coffee (200g)", "Fresh Milk (1L)"]
        }
        
        dish_lower = dish.lower()
        if dish_lower not in recipes:
            logger.warning(f"Recipe not found for: {dish}")
            return f"I don't have a recipe for '{dish}'. Would you like me to suggest something else?"
        
        ingredients = recipes[dish_lower]
        response = f"For {dish}, I'll add:\n"
        added_count = 0
        
        for ingredient in ingredients:
            # Find and add each ingredient - case insensitive search
            item_found = None
            for item in catalog:
                if item['name'].lower() == ingredient.lower():
                    item_found = item
                    break
            
            if item_found:
                if item_found['id'] not in self.cart:
                    self.cart[item_found['id']] = {
                        'name': item_found['name'],
                        'quantity': 1,
                        'price': item_found['price'],
                        'category': item_found['category']
                    }
                    response += f"✓ {item_found['name']}\n"
                    added_count += 1
                else:
                    response += f"✓ {item_found['name']} (already in cart)\n"
            else:
                logger.warning(f"Ingredient '{ingredient}' not found in catalog for recipe '{dish}'")
                response += f"✗ {ingredient} (not in catalog)\n"
        
        logger.info(f"Added {added_count} ingredients for {dish} to cart")
        return response

    @function_tool
    async def place_order(self, customer_name: str = None, delivery_address: str = None):
        """Place the order and save it to a JSON file.
        
        This function finalizes the order by saving cart contents to a JSON file
        with order ID, timestamp, and initial status.
        
        Args:
            customer_name: Optional customer name for the order
            delivery_address: Optional delivery address
        
        Returns:
            Order confirmation message with order ID and total
        """
        if not self.cart:
            return "Your cart is empty! Add some items before placing an order."
        
        grand_total = sum(item['quantity'] * item['price'] for item in self.cart.values())
        
        order_id = str(uuid.uuid4())[:8].upper()
        order = {
            "order_id": order_id,
            "timestamp": datetime.now().isoformat(),
            "customer_name": customer_name or "Guest",
            "delivery_address": delivery_address or "Default Address",
            "items": self.cart,
            "total": grand_total,
            "status": "received",
            "estimated_delivery": "30-45 minutes"
        }
        
        orders_dir = Path(__file__).parent.parent / "shared-data" / "orders"
        orders_dir.mkdir(exist_ok=True)
        
        order_file = orders_dir / f"order_{order_id}.json"
        try:
            with open(order_file, 'w') as f:
                json.dump(order, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save order: {e}")
            return "Sorry, I couldn't save your order. Please try again."
        
        self.cart = {}
        
        logger.info(f"Order placed: {order_id}, Total: ₹{grand_total}")
        return f"Perfect! Your order #{order_id} has been placed with a total of ₹{grand_total}. You can expect delivery in 30-45 minutes. Thank you for ordering with FreshCart!"

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
                style="Conversation",
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

    await session.start(
        agent=FoodOrderingAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
