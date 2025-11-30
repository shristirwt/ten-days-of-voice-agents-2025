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


def load_catalog():
    """Load products catalog from JSON file."""
    catalog_path = Path(__file__).parent.parent / "shared-data" / "day_9_catalog.json"
    logger.info(f"Loading catalog from: {catalog_path}")
    logger.info(f"Catalog file exists: {catalog_path.exists()}")
    try:
        with open(catalog_path, 'r') as f:
            catalog = json.load(f)
            logger.info(f"Successfully loaded {len(catalog)} products from catalog")
            return catalog
    except Exception as e:
        logger.error(f"Error loading catalog: {e}")
        return []


def load_orders():
    """Load orders from JSON file."""
    orders_path = Path(__file__).parent.parent / "shared-data" / "day_9_orders" / "orders.json"
    if os.path.exists(orders_path):
        try:
            with open(orders_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading orders: {e}")
            return []
    return []


def save_orders(orders):
    """Save orders to JSON file."""
    orders_path = Path(__file__).parent.parent / "shared-data" / "day_9_orders" / "orders.json"
    try:
        # Create directory if it doesn't exist
        orders_path.parent.mkdir(parents=True, exist_ok=True)
        with open(orders_path, 'w') as f:
            json.dump(orders, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving orders: {e}")


class EcommerceAgent(Agent):
    """E-commerce voice assistant following ACP-inspired patterns."""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly and helpful e-commerce voice shopping assistant built with Agentic Commerce Protocol (ACP) principles.

YOUR ROLE:
- Help customers browse and search the product catalog by voice
- Maintain a shopping cart throughout the conversation
- Answer questions about products, prices, colors, sizes
- Guide customers through checkout when ready
- Be conversational, friendly, and professional

YOUR CAPABILITIES:
You have access to four main functions:
1. search_products() - Search catalog by category, price, or color
2. add_to_cart() - Add items to shopping cart (does NOT place order yet)
3. get_cart_summary() - Show what's in the cart with total price
4. place_order_from_cart() - Finalize and save the order

YOUR PROCESS:

1. GREETING:
   - Greet warmly: "Hello! Welcome to our shop. I'm your shopping assistant. What can I help you find today? We have mugs, t-shirts, hoodies, and caps."

2. UNDERSTAND THEIR REQUEST:
   - If they ask for products: "Let me search our catalog for you."
   - If they say "Show me all mugs" → call search_products(category="mug")
   - If they say "Do you have t-shirts under 700?" → call search_products(category="t-shirt", max_price=700)
   - If they say "I'm looking for a black hoodie" → call search_products(category="hoodie", color="black")

3. HELP THEM CHOOSE:
   - Ask clarifying questions about colors, sizes, quantity
   - Suggest alternatives if they're interested

4. ADD TO CART (NOT placing order yet):
   - When they want an item: "I'll add that to your cart"
   - Call: add_to_cart(product_id="hoodie-001", quantity=1)
   - They can add multiple items before checkout
   - If they say "I'll take the first one" or "Add 2 of the black hoodies", use add_to_cart()

5. MANAGE THE CART:
   - If they ask "What's in my cart?" → call get_cart_summary() and read it
   - Support adding more items without placing order
   - They build up their cart with multiple items

6. DETECT CHECKOUT INTENT:
   - Listen for phrases like:
     - "That's all"
     - "I'm done"
     - "Place my order"
     - "Check out"
     - "Complete the order"
   - When detected, confirm: "Great! Let me place your order."

7. PLACE THE ORDER:
   - Call: place_order_from_cart(buyer_name="...")
   - Shows: Order ID, number of items, total price, status
   - Announce: "Your order has been saved to our system!"

8. POST-ORDER:
   - If they want more items: "Would you like to continue shopping?"
   - Be helpful and friendly

CRITICAL RULES:
- When user adds items, ALWAYS use add_to_cart() - NOT place_order()
- ONLY call place_order_from_cart() when they explicitly want to checkout
- Let users add multiple items BEFORE finalizing order
- Use exact product IDs from catalog (mug-001, tshirt-002, hoodie-001, cap-002)
- Be conversational but professional
- Keep track of what's being added to cart and confirm each addition

AVAILABLE PRODUCTS:
- Mugs: Stoneware Coffee Mug (₹800), Blue Ceramic Mug (₹950)
- T-Shirts: Classic White (₹599), Black Graphic (₹799)
- Hoodies: Gray Fleece (₹1499), Black Zip (₹1599)
- Caps: Black Baseball (₹499), Red Baseball (₹599)
"""
        )
        self.cart = {}
    
    @function_tool
    async def search_products_tool(self, category: Optional[str] = None, price: Optional[int] = None, color: Optional[str] = None) -> str:
        """Search and list products from the catalog.
        
        This function searches the product catalog for items matching your criteria.
        You can search by category (mug, t-shirt, hoodie, cap), maximum price in INR,
        or color (black, white, gray, blue, red).
        
        Args:
            category: Product category (e.g., "hoodie", "t-shirt", "mug", "cap")
            price: Maximum price in INR (e.g., 700, 1500)
            color: Color filter (e.g., "black", "gray", "white")
        
        Returns:
            A formatted list of matching products with names, descriptions, prices, colors, and sizes
        """
        catalog = load_catalog()
        logger.info(f"[SEARCH] Loaded {len(catalog)} products")
        
        if not catalog:
            return "ERROR: Could not load catalog. No products available."
        
        filtered = catalog
        
        if category:
            search_category = category.lower().replace("-", "").replace(" ", "")
            logger.info(f"[SEARCH] Looking for category: '{category}' (normalized: '{search_category}')")
            logger.info(f"[SEARCH] Catalog categories: {[p['category'] for p in catalog]}")
            filtered = [p for p in filtered if search_category in p["category"].lower().replace("-", "").replace(" ", "")]
            logger.info(f"[SEARCH] Found {len(filtered)} products for category '{category}'")
        
        if price and price > 0:
            filtered = [p for p in filtered if p["price"] <= price]
            logger.info(f"[SEARCH] Filtered to {len(filtered)} products with max price {price}")
        
        if color:
            filtered = [p for p in filtered if p["color"].lower() == color.lower()]
            logger.info(f"[SEARCH] Filtered to {len(filtered)} products with color '{color}'")
        
        if not filtered:
            return "No products found matching your criteria. Please try different filters."
        
        result = f"Found {len(filtered)} product(s):\n\n"
        for idx, product in enumerate(filtered, 1):
            result += f"{idx}. {product['name']}\n"
            result += f"   Description: {product['description']}\n"
            result += f"   Price: ₹{product['price']} {product['currency']}\n"
            result += f"   Category: {product['category']}\n"
            result += f"   Color: {product['color']}\n"
            result += f"   Size: {product['size']}\n\n"
        
        logger.info(f"[SEARCH] Returning {len(filtered)} products")
        return result
    
    @function_tool
    async def add_to_cart_tool(
        self,
        product_id: str,
        quantity: int = 1,
    ) -> str:
        """Add an item to the shopping cart with specified quantity.
        
        This function adds a product from the catalog to your shopping cart.
        You must provide the product ID (e.g., "hoodie-001", "mug-002", "tshirt-001", "cap-002")
        and optionally specify quantity (default is 1).
        
        Args:
            product_id: The product ID from catalog (e.g., "hoodie-001", "mug-001")
            quantity: How many to add (default 1)
        
        Returns:
            Confirmation message with item name, quantity, and price
        """
        catalog = load_catalog()
        logger.info(f"[CART] Adding {quantity} of product_id: {product_id}")
        
        product = next((p for p in catalog if p["id"] == product_id), None)
        
        if not product:
            logger.warning(f"[CART] Product not found: {product_id}")
            return f"Product with ID '{product_id}' not found."
        
        if quantity < 1:
            return "Quantity must be at least 1."
        
        if product_id in self.cart:
            self.cart[product_id]['quantity'] += quantity
        else:
            self.cart[product_id] = {
                'name': product['name'],
                'quantity': quantity,
                'price': product['price'],
                'category': product['category']
            }
        
        total_price = self.cart[product_id]['quantity'] * self.cart[product_id]['price']
        logger.info(f"[CART] Added to cart. New total for item: ₹{total_price}")
        return f"Great! I've added {quantity} {product['name']} to your cart at ₹{product['price']} each."
    
    @function_tool
    async def get_cart_summary_tool(self) -> str:
        """Get a summary of all items currently in the shopping cart.
        
        This function shows you everything in your cart including product names,
        quantities, unit prices for each item, and the grand total.
        If the cart is empty, it will tell you so.
        
        Returns:
            Formatted string with all items, quantities, prices, and total
        """
        if not self.cart:
            return "Your cart is empty. Start shopping!"
        
        summary = "📋 Your Cart:\n\n"
        grand_total = 0
        
        for idx, (product_id, item) in enumerate(self.cart.items(), 1):
            item_total = item['quantity'] * item['price']
            grand_total += item_total
            summary += f"{idx}. {item['name']} (x{item['quantity']}) @ ₹{item['price']} each = ₹{item_total}\n"
        
        summary += f"\n💰 Grand Total: ₹{grand_total}\n"
        logger.info(f"[CART] Showing cart with {len(self.cart)} items, total: ₹{grand_total}")
        return summary
    
    @function_tool
    async def place_order_tool(
        self,
        buyer_name: Optional[str] = "Customer",
    ) -> str:
        """Finalize and place the order from your shopping cart.
        
        This function takes everything in your cart and creates a final order.
        The order is saved permanently to our system with your name, all items,
        total price, and timestamp. Your cart is then cleared.
        
        Args:
            buyer_name: Your name (default is "Customer" if not provided)
        
        Returns:
            Order confirmation with order ID, all items, and total amount
        """
        if not self.cart:
            return "Your cart is empty. Please add items before placing an order."
        
        logger.info(f"[ORDER] Placing order for {buyer_name} with {len(self.cart)} items")
        
        items = []
        grand_total = 0
        
        for product_id, item in self.cart.items():
            item_total = item['quantity'] * item['price']
            grand_total += item_total
            
            items.append({
                "product_id": product_id,
                "product_name": item['name'],
                "quantity": item['quantity'],
                "unit_amount": item['price'],
                "currency": "INR",
            })
        
        order = {
            "id": f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "buyer_name": buyer_name,
            "items": items,
            "total": grand_total,
            "currency": "INR",
            "created_at": datetime.now().isoformat(),
            "status": "CONFIRMED",
        }
        
        orders = load_orders()
        orders.append(order)
        save_orders(orders)
        
        confirmation = (
            f"✓ Order confirmed!\n"
            f"Order ID: {order['id']}\n"
            f"Items: {len(items)} product(s)\n"
            f"Total: ₹{grand_total} INR\n"
            f"Buyer: {buyer_name}\n"
            f"Status: {order['status']}\n"
            f"Your order has been saved to our system!"
        )
        
        logger.info(f"[ORDER] Order placed: ID={order['id']}, Total=₹{grand_total}")
        
        # Clear the cart after placing order
        self.cart = {}
        return confirmation



def prewarm(proc: JobProcess):
    """Warm up resources on worker process startup."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the e-commerce agent."""
    logger.info("Starting E-commerce Voice Agent")
    
    agent = EcommerceAgent()
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Narration",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        preemptive_generation=True,
    )
    
    await session.start(agent=agent, room=ctx.room)
    
    logger.info("E-commerce agent ready for conversation")
    
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
