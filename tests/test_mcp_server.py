import asyncio
from fastmcp import Client

client = Client("http://localhost:9101/mcp")

async def call_tool(name: str):
    async with client:
        result = await client.call_tool("extract_keywords", {"answer": name})
        print(result)

asyncio.run(call_tool("Revenue and profit increased this quarter due to strong sales and reduced expenses."))
