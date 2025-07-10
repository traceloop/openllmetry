import os
import asyncio
from llama_parse import LlamaParse
from traceloop.sdk import Traceloop

Traceloop.init()

resource_url = "https://arxiv.org/pdf/1706.03762.pdf"


async def demo_core_parsing():
    """Demonstrate core LlamaParse functionality with instrumentation"""
    print("=== Core LlamaParse Methods ===")

    print("1. aload_data() - markdown format")
    parser1 = LlamaParse(result_type="markdown")
    result1 = await parser1.aload_data(resource_url)
    print(f"   Parsed {len(result1)} documents (markdown)")

    print("2. aload_data() - text format")
    parser2 = LlamaParse(result_type="text")
    result2 = await parser2.aload_data(resource_url)
    print(f"   Parsed {len(result2)} documents (text)")

    print("3. aload_data() - with custom instructions")
    parser3 = LlamaParse(
        result_type="markdown",
        parsing_instruction="Focus on extracting mathematical formulas and technical terms"
    )
    result3 = await parser3.aload_data(resource_url)
    print(f"   Parsed {len(result3)} documents (custom instructions)")

    """Demonstrate advanced features that may require specific parameters"""
    print("\n=== Advanced Features ===")

    print("Testing other instrumented methods:")

    download_path = "./tmp_downloads"
    os.makedirs(download_path, exist_ok=True)

    # Test aget_json with a parser that has already loaded data
    try:
        parser = LlamaParse(result_type="markdown")
        json_result = await parser.aget_json(resource_url)
        print(f"   aget_json() succeeded: {len(json_result) if json_result else 0} items")
    except Exception as e:
        print(f"   aget_json() failed: {str(e)[:50]}...")

    # Test aget_images
    try:
        image_parser = LlamaParse(result_type="markdown", extract_images=True)
        json_result = await image_parser.aget_json(resource_url)
        images = await image_parser.aget_images(json_result, download_path)
        print(f"   aget_images() succeeded: {len(images) if images else 0} images")
    except Exception as e:
        print(f"   aget_images() failed: {str(e)[:50]}...")

    # Test aget_charts
    try:
        chart_parser = LlamaParse(result_type="markdown", extract_charts=True)
        json_result = await chart_parser.aget_json(resource_url)
        charts = await chart_parser.aget_charts(json_result, download_path)
        print(f"   aget_charts() succeeded: {len(charts) if charts else 0} charts")
    except Exception as e:
        print(f"   aget_charts() failed: {str(e)[:50]}...")


async def main():
    try:
        await demo_core_parsing()

        print("\nDemo complete!")

    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
