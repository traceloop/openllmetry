import inspect
import sys

def debug_stack():
    """Debug function to examine stack frames"""
    current_frame = inspect.currentframe()
    frame_count = 0

    print("=== STACK FRAME ANALYSIS ===")
    while current_frame and frame_count < 10:  # Limit to avoid infinite loops
        frame_info = inspect.getframeinfo(current_frame)
        print(f"Frame {frame_count}:")
        print(f"  File: {frame_info.filename}")
        print(f"  Function: {frame_info.function}")
        print(f"  Line: {frame_info.lineno}")

        # Show local variables
        print("  Local variables:")
        for name, value in current_frame.f_locals.items():
            if hasattr(value, '__class__'):
                print(f"    {name}: {value.__class__.__name__}")
                if value.__class__.__name__ == 'FastMCP':
                    print(f"      -> FastMCP name: {getattr(value, 'name', 'NO NAME')}")
            else:
                print(f"    {name}: {type(value).__name__}")

        print("---")
        current_frame = current_frame.f_back
        frame_count += 1

    print("=== END STACK ANALYSIS ===")

if __name__ == "__main__":
    debug_stack()