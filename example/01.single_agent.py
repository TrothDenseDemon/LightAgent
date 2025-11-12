import asyncio

from LightAgent import LightAgent


async def main() -> None:
    """Run LightAgent in asynchronous mode."""
    agent = LightAgent(model="gpt-4o-mini", api_key="your_api_key", base_url="http://your_base_url/v1")
    response = await agent.arun("Hello, who are you?")
    print(response)


if __name__ == "__main__":
    # Wrap the async entry point with asyncio.run to stay compatible with synchronous scripts.
    asyncio.run(main())