import asyncio
import random
import argparse
import time
import csv
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
from traceloop.sdk import Traceloop

from agents import Agent, function_tool, RunContextWrapper, Runner, ToolCallOutputItem
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseOutputItemAddedEvent,
    ResponseFunctionToolCall,
    ResponseOutputText,
    ResponseOutputRefusal,
    ResponseFunctionCallArgumentsDoneEvent,
)

load_dotenv()

Traceloop.init(
    app_name="ecommerce-agent-demo",
    disable_batch=False,
)


# =============================================================================
# Pydantic Models for Tool Responses
# =============================================================================

class ProductInfo(BaseModel):
    id: int
    title: str
    description: str
    price: float
    discount_percentage: float = 0.0
    rating: float = 0.0
    stock: int = 0
    brand: str = ""
    category: str = ""
    thumbnail: str = ""


class ProductSearchResponse(BaseModel):
    status: str
    message: str
    products: List[ProductInfo] = []
    total: int = 0


class ProductDetailsResponse(BaseModel):
    status: str
    message: str
    product: Optional[ProductInfo] = None


class CategoryListResponse(BaseModel):
    status: str
    message: str
    categories: List[str] = []


class CategoryProductsResponse(BaseModel):
    status: str
    message: str
    category: str
    products: List[ProductInfo] = []
    total: int = 0


class CartItem(BaseModel):
    product_id: int
    product_title: str
    quantity: int
    price: float
    total: float


class CartResponse(BaseModel):
    status: str
    message: str
    cart_id: int = 0
    items: List[CartItem] = []
    total_items: int = 0
    total_price: float = 0.0


class AddToCartResponse(BaseModel):
    status: str
    message: str
    cart_id: int = 0
    added_product: Optional[CartItem] = None


class CouponResponse(BaseModel):
    status: str
    message: str
    coupon_code: str = ""
    discount_percent: float = 0.0
    applied: bool = False


class CheckoutResponse(BaseModel):
    status: str
    message: str
    order_id: str = ""
    items_count: int = 0
    subtotal: float = 0.0
    discount: float = 0.0
    total: float = 0.0
    payment_status: str = ""


# =============================================================================
# Context Dataclass
# =============================================================================

@dataclass
class EcommerceContext:
    """Context for the e-commerce agent application."""
    user_id: int = 1
    cart_id: int = 0
    cart_items: List[Dict] = field(default_factory=list)
    applied_coupons: List[str] = field(default_factory=list)
    cart_total: float = 0.0


# =============================================================================
# Mock Data for Coupons
# =============================================================================

VALID_COUPONS = {
    "SAVE10": 10.0,
    "SAVE20": 20.0,
    "WELCOME15": 15.0,
    "FLASH25": 25.0,
    "VIP30": 30.0,
}


# =============================================================================
# Tool Definitions (8 tools)
# =============================================================================

@function_tool
async def search_products(
    cw: RunContextWrapper[EcommerceContext],
    query: str,
    limit: int = 10
) -> ProductSearchResponse:
    """
    Search for products by keyword using DummyJSON API.

    Args:
        query: Search keyword (e.g., "laptop", "phone", "shirt")
        limit: Maximum number of results to return (default: 10)

    Returns:
        List of products matching the search query
    """
    print(f"Searching products for: '{query}' (limit: {limit})")

    try:
        await asyncio.sleep(0.3)

        url = "https://dummyjson.com/products/search"
        params = {"q": query, "limit": limit}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        products = []

        for p in data.get("products", []):
            product = ProductInfo(
                id=p.get("id", 0),
                title=p.get("title", ""),
                description=p.get("description", ""),
                price=p.get("price", 0.0),
                discount_percentage=p.get("discountPercentage", 0.0),
                rating=p.get("rating", 0.0),
                stock=p.get("stock", 0),
                brand=p.get("brand", ""),
                category=p.get("category", ""),
                thumbnail=p.get("thumbnail", "")
            )
            products.append(product)

        return ProductSearchResponse(
            status="success",
            message=f"Found {len(products)} products for '{query}'",
            products=products,
            total=len(products)
        )

    except requests.RequestException as e:
        return ProductSearchResponse(
            status="error",
            message=f"Failed to search products: {str(e)}"
        )


@function_tool
async def get_product_details(
    cw: RunContextWrapper[EcommerceContext],
    product_id: int
) -> ProductDetailsResponse:
    """
    Get detailed information about a specific product.

    Args:
        product_id: The unique ID of the product

    Returns:
        Detailed product information including price, stock, and description
    """
    print(f"Getting details for product ID: {product_id}")

    try:
        await asyncio.sleep(0.3)

        url = f"https://dummyjson.com/products/{product_id}"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        p = response.json()
        product = ProductInfo(
            id=p.get("id", 0),
            title=p.get("title", ""),
            description=p.get("description", ""),
            price=p.get("price", 0.0),
            discount_percentage=p.get("discountPercentage", 0.0),
            rating=p.get("rating", 0.0),
            stock=p.get("stock", 0),
            brand=p.get("brand", ""),
            category=p.get("category", ""),
            thumbnail=p.get("thumbnail", "")
        )

        return ProductDetailsResponse(
            status="success",
            message=f"Retrieved details for '{product.title}'",
            product=product
        )

    except requests.RequestException as e:
        return ProductDetailsResponse(
            status="error",
            message=f"Failed to get product details: {str(e)}"
        )


@function_tool
async def get_categories(
    cw: RunContextWrapper[EcommerceContext]
) -> CategoryListResponse:
    """
    Get all available product categories.

    Returns:
        List of all product categories in the store
    """
    print("Getting all product categories")

    try:
        await asyncio.sleep(0.3)

        url = "https://dummyjson.com/products/categories"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        # DummyJSON returns list of category objects with slug and name
        categories = []
        for cat in data:
            if isinstance(cat, dict):
                categories.append(cat.get("slug", cat.get("name", "")))
            else:
                categories.append(str(cat))

        return CategoryListResponse(
            status="success",
            message=f"Found {len(categories)} categories",
            categories=categories
        )

    except requests.RequestException as e:
        return CategoryListResponse(
            status="error",
            message=f"Failed to get categories: {str(e)}"
        )


@function_tool
async def get_products_by_category(
    cw: RunContextWrapper[EcommerceContext],
    category: str,
    limit: int = 10
) -> CategoryProductsResponse:
    """
    Get products in a specific category.

    Args:
        category: The category slug (e.g., "smartphones", "laptops", "furniture")
        limit: Maximum number of products to return (default: 10)

    Returns:
        List of products in the specified category
    """
    print(f"Getting products in category: '{category}' (limit: {limit})")

    try:
        await asyncio.sleep(0.3)

        url = f"https://dummyjson.com/products/category/{category}"
        params = {"limit": limit}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        products = []

        for p in data.get("products", []):
            product = ProductInfo(
                id=p.get("id", 0),
                title=p.get("title", ""),
                description=p.get("description", ""),
                price=p.get("price", 0.0),
                discount_percentage=p.get("discountPercentage", 0.0),
                rating=p.get("rating", 0.0),
                stock=p.get("stock", 0),
                brand=p.get("brand", ""),
                category=p.get("category", ""),
                thumbnail=p.get("thumbnail", "")
            )
            products.append(product)

        return CategoryProductsResponse(
            status="success",
            message=f"Found {len(products)} products in '{category}'",
            category=category,
            products=products,
            total=len(products)
        )

    except requests.RequestException as e:
        return CategoryProductsResponse(
            status="error",
            message=f"Failed to get category products: {str(e)}",
            category=category
        )


@function_tool
async def add_to_cart(
    cw: RunContextWrapper[EcommerceContext],
    product_id: int,
    quantity: int = 1
) -> AddToCartResponse:
    """
    Add a product to the shopping cart.

    Args:
        product_id: The ID of the product to add
        quantity: Number of items to add (default: 1)

    Returns:
        Confirmation of the item added to cart
    """
    print(f"Adding product {product_id} to cart (quantity: {quantity})")

    try:
        await asyncio.sleep(0.3)

        # First get product details
        url = f"https://dummyjson.com/products/{product_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        product = response.json()

        # Add to DummyJSON cart API
        cart_url = "https://dummyjson.com/carts/add"
        cart_data = {
            "userId": cw.context.user_id,
            "products": [{"id": product_id, "quantity": quantity}]
        }

        cart_response = requests.post(cart_url, json=cart_data, timeout=10)
        cart_response.raise_for_status()

        cart_result = cart_response.json()

        # Update context
        item_total = product.get("price", 0) * quantity
        cart_item = CartItem(
            product_id=product_id,
            product_title=product.get("title", "Unknown"),
            quantity=quantity,
            price=product.get("price", 0),
            total=item_total
        )

        cw.context.cart_items.append({
            "product_id": product_id,
            "title": product.get("title", ""),
            "quantity": quantity,
            "price": product.get("price", 0),
            "total": item_total
        })
        cw.context.cart_total += item_total
        cw.context.cart_id = cart_result.get("id", 1)

        return AddToCartResponse(
            status="success",
            message=f"Added {quantity}x '{product.get('title', 'product')}' to cart",
            cart_id=cw.context.cart_id,
            added_product=cart_item
        )

    except requests.RequestException as e:
        return AddToCartResponse(
            status="error",
            message=f"Failed to add to cart: {str(e)}"
        )


@function_tool
async def get_cart(
    cw: RunContextWrapper[EcommerceContext]
) -> CartResponse:
    """
    Get the current shopping cart contents.

    Returns:
        Current cart contents including items, quantities, and totals
    """
    print(f"Getting cart for user {cw.context.user_id}")

    try:
        await asyncio.sleep(0.3)

        # Use context cart if we have items, otherwise fetch from API
        if cw.context.cart_items:
            items = []
            for item in cw.context.cart_items:
                cart_item = CartItem(
                    product_id=item["product_id"],
                    product_title=item["title"],
                    quantity=item["quantity"],
                    price=item["price"],
                    total=item["total"]
                )
                items.append(cart_item)

            return CartResponse(
                status="success",
                message=f"Cart has {len(items)} items",
                cart_id=cw.context.cart_id,
                items=items,
                total_items=sum(item["quantity"] for item in cw.context.cart_items),
                total_price=cw.context.cart_total
            )

        # Fetch from DummyJSON
        url = f"https://dummyjson.com/carts/user/{cw.context.user_id}"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        carts = data.get("carts", [])

        if not carts:
            return CartResponse(
                status="success",
                message="Cart is empty",
                cart_id=0,
                items=[],
                total_items=0,
                total_price=0.0
            )

        cart = carts[0]
        items = []
        for p in cart.get("products", []):
            item = CartItem(
                product_id=p.get("id", 0),
                product_title=p.get("title", ""),
                quantity=p.get("quantity", 0),
                price=p.get("price", 0),
                total=p.get("total", 0)
            )
            items.append(item)

        return CartResponse(
            status="success",
            message=f"Cart has {len(items)} items",
            cart_id=cart.get("id", 0),
            items=items,
            total_items=cart.get("totalQuantity", 0),
            total_price=cart.get("total", 0.0)
        )

    except requests.RequestException as e:
        return CartResponse(
            status="error",
            message=f"Failed to get cart: {str(e)}"
        )


@function_tool
async def apply_coupon(
    cw: RunContextWrapper[EcommerceContext],
    coupon_code: str
) -> CouponResponse:
    """
    Apply a discount coupon to the cart.

    Args:
        coupon_code: The coupon code to apply (e.g., "SAVE10", "WELCOME15")

    Returns:
        Confirmation of coupon application and discount amount
    """
    print(f"Applying coupon: '{coupon_code}'")

    await asyncio.sleep(0.2)

    code_upper = coupon_code.upper()

    if code_upper in cw.context.applied_coupons:
        return CouponResponse(
            status="error",
            message=f"Coupon '{coupon_code}' has already been applied",
            coupon_code=coupon_code,
            discount_percent=0.0,
            applied=False
        )

    if code_upper in VALID_COUPONS:
        discount = VALID_COUPONS[code_upper]
        cw.context.applied_coupons.append(code_upper)

        return CouponResponse(
            status="success",
            message=f"Coupon '{coupon_code}' applied! {discount}% discount",
            coupon_code=coupon_code,
            discount_percent=discount,
            applied=True
        )
    else:
        return CouponResponse(
            status="error",
            message=f"Invalid coupon code: '{coupon_code}'",
            coupon_code=coupon_code,
            discount_percent=0.0,
            applied=False
        )


@function_tool
async def checkout(
    cw: RunContextWrapper[EcommerceContext],
    payment_method: str = "credit_card"
) -> CheckoutResponse:
    """
    Complete the purchase and checkout.

    Args:
        payment_method: Payment method to use ("credit_card", "paypal", "apple_pay")

    Returns:
        Order confirmation with total, discounts, and order ID
    """
    print(f"Processing checkout with payment method: '{payment_method}'")

    await asyncio.sleep(0.3)

    # Calculate totals
    subtotal = cw.context.cart_total
    if subtotal == 0:
        # Use sample total if cart was fetched from API
        subtotal = 100.0

    # Calculate discount from coupons
    total_discount_percent = sum(
        VALID_COUPONS.get(code, 0) for code in cw.context.applied_coupons
    )
    total_discount_percent = min(total_discount_percent, 50.0)  # Max 50% discount

    discount_amount = subtotal * (total_discount_percent / 100)
    final_total = subtotal - discount_amount

    # Generate order ID
    order_id = f"ORD-{random.randint(100000, 999999)}"

    items_count = len(cw.context.cart_items) if cw.context.cart_items else 1

    return CheckoutResponse(
        status="success",
        message=f"Order {order_id} placed successfully!",
        order_id=order_id,
        items_count=items_count,
        subtotal=round(subtotal, 2),
        discount=round(discount_amount, 2),
        total=round(final_total, 2),
        payment_status="paid"
    )


# =============================================================================
# Dataset Collector
# =============================================================================

class DatasetCollector:
    """Collects tool call data for BFCL-style dataset generation."""

    def __init__(self):
        self.rows: List[Dict[str, Any]] = []
        self.current_query_id: int = 0
        self.current_turn: int = 0
        self.current_history: List[Dict[str, Any]] = []
        self.current_query: str = ""

    def start_new_query(self, query: str):
        """Start tracking a new query."""
        self.current_query = query
        self.current_turn = 0
        self.current_history = []

    def record_tool_call(
        self,
        tool_name: str,
        tool_arguments: Dict[str, Any],
        result_status: str = "pass"
    ):
        """Record a single tool call."""
        row = {
            "id": f"ecommerce_{self.current_query_id}_turn{self.current_turn}",
            "category": "ecommerce_multi_turn",
            "user_query": self.current_query,
            "selected_tool": tool_name,
            "selected_tool_arguments": json.dumps(tool_arguments),
            "expected_result": result_status,
            "failure_mode": "",
            "expected_score_min": 0.75,
            "expected_score_max": 1.0,
            "has_agent_history": len(self.current_history) > 0,
            "agent_history": json.dumps(self.current_history),
            "available_tools": json.dumps(self._get_available_tools_schema())
        }
        self.rows.append(row)

        # Update history for next turn
        self.current_history.append({
            "tool_name": tool_name,
            "arguments": tool_arguments,
            "result": {"success": result_status == "pass"}
        })
        self.current_turn += 1

    def end_query(self):
        """End tracking for current query."""
        self.current_query_id += 1

    def export_to_csv(self, filepath: str):
        """Export collected data to CSV file."""
        if not self.rows:
            print("No data to export")
            return

        fieldnames = [
            "id", "category", "user_query", "selected_tool",
            "selected_tool_arguments", "expected_result", "failure_mode",
            "expected_score_min", "expected_score_max", "has_agent_history",
            "agent_history", "available_tools"
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

        print(f"Exported {len(self.rows)} rows to {filepath}")

    def _get_available_tools_schema(self) -> List[Dict[str, Any]]:
        """Get JSON schema for all available tools."""
        return [
            {
                "name": "search_products",
                "description": "Search for products by keyword using DummyJSON API.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search keyword"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_product_details",
                "description": "Get detailed information about a specific product.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "integer",
                            "description": "The unique ID of the product"
                        }
                    },
                    "required": ["product_id"]
                }
            },
            {
                "name": "get_categories",
                "description": "Get all available product categories.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_products_by_category",
                "description": "Get products in a specific category.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Category slug"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10
                        }
                    },
                    "required": ["category"]
                }
            },
            {
                "name": "add_to_cart",
                "description": "Add a product to the shopping cart.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "integer",
                            "description": "Product ID to add"
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "Quantity to add",
                            "default": 1
                        }
                    },
                    "required": ["product_id"]
                }
            },
            {
                "name": "get_cart",
                "description": "Get the current shopping cart contents.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "apply_coupon",
                "description": "Apply a discount coupon to the cart.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "coupon_code": {
                            "type": "string",
                            "description": "Coupon code to apply"
                        }
                    },
                    "required": ["coupon_code"]
                }
            },
            {
                "name": "checkout",
                "description": "Complete the purchase and checkout.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "payment_method": {
                            "type": "string",
                            "description": "Payment method",
                            "default": "credit_card"
                        }
                    },
                    "required": []
                }
            }
        ]


# =============================================================================
# E-commerce Agent
# =============================================================================

class EcommerceAgent(Agent[EcommerceContext]):
    """Specialized agent for e-commerce with 8 tools, always completing checkout."""

    def __init__(self, model: str = "gpt-4o"):
        super().__init__(
            name="E-commerce Shopping Agent",
            instructions="""
            You are an expert e-commerce shopping assistant. Your PRIMARY GOAL is to help
            users find products, manage their cart, and complete purchases.

            Your workflow:
            1. Understand what the user wants to buy or browse
            2. Search for products or browse categories as needed
            3. Get product details if the user wants more info
            4. Add items to cart when requested
            5. Apply coupons if provided
            6. ALWAYS end by completing checkout when the user wants to buy

            Your 8 tools:
            1. search_products - Search products by keyword
            2. get_product_details - Get detailed product info
            3. get_categories - List available categories
            4. get_products_by_category - Browse products in a category
            5. add_to_cart - Add product to shopping cart
            6. get_cart - View current cart contents
            7. apply_coupon - Apply discount codes (SAVE10, SAVE20, WELCOME15, FLASH25, VIP30)
            8. checkout - Complete the purchase (REQUIRED for buy requests!)

            Response patterns based on request type:

            SEARCH REQUESTS ("find me a laptop", "looking for phones"):
            - search_products → get_product_details (optional) → add_to_cart → checkout

            BROWSE REQUESTS ("what categories do you have", "show me electronics"):
            - get_categories → get_products_by_category → get_product_details → add_to_cart → checkout

            CART REQUESTS ("what's in my cart", "check my order"):
            - get_cart → checkout

            DISCOUNT REQUESTS ("I have code SAVE10", "apply my coupon"):
            - apply_coupon → checkout

            FULL PURCHASE ("buy the iPhone with discount SAVE20"):
            - search_products → add_to_cart → apply_coupon → checkout

            CRITICAL: Every purchase request must end with checkout. If the user says
            "buy", "purchase", "order", or similar, you MUST call checkout at the end.

            When searching, pick the most relevant product and proceed. Don't ask for
            clarification unless absolutely necessary.
            """,
            model=model,
            tools=[
                search_products,
                get_product_details,
                get_categories,
                get_products_by_category,
                add_to_cart,
                get_cart,
                apply_coupon,
                checkout
            ],
        )


# =============================================================================
# Stream Handler with Dataset Collection
# =============================================================================

async def handle_runner_stream(
    runner: "Runner",
    collector: Optional[DatasetCollector] = None
):
    """Process runner events and collect tool call data."""

    tool_calls_made = []
    response_text_parts = []
    # Track pending tool calls by item_id to capture arguments when complete
    pending_tool_calls: Dict[str, str] = {}  # item_id -> tool_name

    async for event in runner.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
                response_text_parts.append(event.data.delta)
            elif isinstance(event.data, ResponseOutputItemAddedEvent):
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    tool_name = event.data.item.name
                    item_id = getattr(event.data.item, "id", None)
                    tool_calls_made.append(tool_name)
                    print(f"\n[Calling tool: {tool_name}]")

                    # Track this tool call to capture arguments later
                    if item_id:
                        pending_tool_calls[item_id] = tool_name

            elif isinstance(event.data, ResponseFunctionCallArgumentsDoneEvent):
                # Arguments are now complete - record to dataset collector
                item_id = getattr(event.data, "item_id", None)
                arguments_str = getattr(event.data, "arguments", "{}")

                tool_name = pending_tool_calls.get(item_id, "unknown")

                try:
                    tool_arguments = json.loads(arguments_str) if arguments_str else {}
                except json.JSONDecodeError:
                    tool_arguments = {}

                # Record to dataset collector with complete arguments
                if collector:
                    collector.record_tool_call(
                        tool_name=tool_name,
                        tool_arguments=tool_arguments,
                        result_status="pass"
                    )

        elif event.type == "run_item_stream_event":
            if event.name == "tool_output" and isinstance(
                event.item, ToolCallOutputItem
            ):
                raw_item = event.item.raw_item
                content = (
                    raw_item.get("content")
                    if isinstance(raw_item, dict)
                    else getattr(raw_item, "content", "")
                )
                if content:
                    preview = str(content)[:150]
                    print(f"[Tool output: {preview}...]", end="", flush=True)

            elif event.name == "message_output_created":
                raw_item = event.item.raw_item
                role = getattr(raw_item, "role", None)
                if role is None and isinstance(raw_item, dict):
                    role = raw_item.get("role")

                if role == "assistant":
                    content_parts = []
                    for part in getattr(raw_item, "content", []):
                        if isinstance(part, ResponseOutputText):
                            content_parts.append(part.text)
                            response_text_parts.append(part.text)
                        elif isinstance(part, ResponseOutputRefusal):
                            content_parts.append(part.refusal)
                            response_text_parts.append(part.refusal)
                    if content_parts:
                        print("".join(content_parts), end="", flush=True)

    print()
    return tool_calls_made, "".join(response_text_parts)


# =============================================================================
# Query Runner
# =============================================================================

async def run_ecommerce_query(
    query: str,
    collector: Optional[DatasetCollector] = None,
    return_response_text: bool = False
):
    """
    Run a single e-commerce query.

    Args:
        query: The shopping query
        collector: Optional DatasetCollector for recording tool calls
        return_response_text: If True, returns response text

    Returns:
        Either response_text (str) or tool_calls (list)
    """

    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    if collector:
        collector.start_new_query(query)

    ecommerce_agent = EcommerceAgent()

    print("\nAgent Response: ", end="", flush=True)

    messages = [{"role": "user", "content": query}]
    runner = Runner().run_streamed(starting_agent=ecommerce_agent, input=messages)
    tool_calls, response_text = await handle_runner_stream(runner, collector)

    if collector:
        collector.end_query()

    print(f"\n{'='*80}")
    print(f"Query completed! Tools used: {', '.join(tool_calls) if tool_calls else 'None'}")
    print(f"{'='*80}\n")

    if return_response_text:
        return response_text
    else:
        return tool_calls


# =============================================================================
# Query Generation
# =============================================================================

def generate_ecommerce_queries(n: int = 10) -> List[str]:
    """Generate diverse e-commerce queries with varying tool depth (2-5 tools)."""

    templates = [
        # === SIMPLE QUERIES (2-3 tools) ===
        # search_products → checkout
        "Find me a {product}",
        "I want to buy a {product}",
        "Search for {product}",
        "Looking for a good {product}",

        # get_cart → checkout
        "What's in my cart?",
        "Show me my shopping cart",
        "Check my current order",

        # get_categories → get_products_by_category
        "What categories do you have?",
        "Show me all the product categories",
        "What can I browse?",

        # === MEDIUM QUERIES (3-4 tools) ===
        # search_products → get_product_details → add_to_cart
        "Find a {product} and show me the details",
        "Search for {product} and add it to my cart",
        "I need a {product}, find one and add it to cart",

        # get_categories → get_products_by_category → add_to_cart
        "Show me {category} products and add the best one to cart",
        "Browse {category} and pick something for me",

        # search_products → add_to_cart → apply_coupon
        "Find a {product} and apply code {coupon}",
        "I want a {product}, use my discount code {coupon}",

        # === COMPLEX QUERIES (4-5 tools) ===
        # search_products → get_product_details → add_to_cart → apply_coupon → checkout
        "Find the best {product}, check its details, add to cart with code {coupon}, and checkout",
        "Search for {product}, get details, add it to my cart, apply {coupon}, and complete purchase",

        # get_categories → get_products_by_category → get_product_details → add_to_cart → checkout
        "Browse {category}, show me the top product details, add to cart, and checkout",
        "What's in {category}? Pick the best one, add to cart, and buy it",

        # Full flow with multiple products
        "I need a {product} from {category}, show details, add to cart with {coupon}, checkout",
        "Find me a {product}, apply discount {coupon}, and complete my order",

        # === COMPARISON/RESEARCH QUERIES ===
        "Compare {product} options and buy the best one",
        "What {product} do you recommend? Add it to cart and checkout",
        "Show me {category} options, pick the best value, and purchase",

        # === SPECIFIC SCENARIOS ===
        "I have coupon {coupon}, find a {product} and use it",
        "Buy a {product} for under ${price}",
        "Get me the cheapest {product} and checkout",
        "Find a highly rated {product} and purchase it",
    ]

    products = [
        "laptop", "smartphone", "headphones", "tablet", "watch",
        "camera", "speaker", "keyboard", "mouse", "monitor",
        "TV", "shoes", "bag", "sunglasses", "perfume",
        "skincare product", "dress", "jacket", "furniture", "lamp"
    ]

    categories = [
        "smartphones", "laptops", "fragrances", "skincare",
        "groceries", "home-decoration", "furniture", "tops",
        "womens-dresses", "womens-shoes", "mens-shirts", "mens-shoes",
        "mens-watches", "womens-watches", "womens-bags", "womens-jewellery",
        "sunglasses", "automotive", "motorcycle", "lighting"
    ]

    coupons = list(VALID_COUPONS.keys())

    prices = ["50", "100", "200", "500", "1000"]

    queries = []
    for _ in range(n):
        template = random.choice(templates)

        query = template.format(
            product=random.choice(products),
            category=random.choice(categories),
            coupon=random.choice(coupons),
            price=random.choice(prices)
        )
        queries.append(query)

    return queries


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Main entry point for the e-commerce agent application."""

    parser = argparse.ArgumentParser(description="E-commerce Agent Demo")
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of queries to run (default: 3)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between queries in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ecommerce_dataset.csv",
        help="Output CSV file path (default: ecommerce_dataset.csv)"
    )
    parser.add_argument(
        "--no-collect",
        action="store_true",
        help="Disable dataset collection"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("E-commerce Agent with OpenAI Agents SDK")
    print("=" * 80)
    print(f"Running {args.count} shopping queries...")
    print("Goal: Complete purchases with varying tool trajectories (2-5 tools)")
    print("Using DummyJSON API for products, categories, and carts")
    print("8 Tools: search, details, categories, category_products, add_to_cart, get_cart, coupon, checkout")
    print("=" * 80)
    print()

    # Initialize dataset collector
    collector = None if args.no_collect else DatasetCollector()

    queries = generate_ecommerce_queries(args.count)

    all_tool_calls = []
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Query {i} of {args.count}")
        print(f"{'#'*80}\n")

        tool_calls = await run_ecommerce_query(query, collector=collector)
        all_tool_calls.append({
            "query": query,
            "tools_used": tool_calls,
            "tool_count": len(tool_calls)
        })

        if i < args.count:
            print(f"\nWaiting {args.delay} seconds before next query...")
            time.sleep(args.delay)

    # Export dataset
    if collector:
        collector.export_to_csv(args.output)

    # Summary
    print("\n\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total queries executed: {len(all_tool_calls)}")

    tool_usage = {}
    for result in all_tool_calls:
        for tool in result["tools_used"]:
            tool_usage[tool] = tool_usage.get(tool, 0) + 1

    print("\nTool usage statistics:")
    for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {tool}: {count} times")

    print("\nTrajectory variation:")
    unique_trajectories = len(set(tuple(r["tools_used"]) for r in all_tool_calls))
    print(f"  - Unique tool call sequences: {unique_trajectories}/{len(all_tool_calls)}")

    avg_tools = sum(r["tool_count"] for r in all_tool_calls) / len(all_tool_calls) if all_tool_calls else 0
    print(f"  - Average tools per query: {avg_tools:.2f}")

    if collector:
        print(f"\nDataset exported to: {args.output}")
        print(f"Total rows in dataset: {len(collector.rows)}")

    print("\n" + "=" * 80)
    print("E-commerce Agent demo completed!")
    print("All spans captured by OpenTelemetry instrumentation")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
