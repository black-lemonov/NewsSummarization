import asyncio

from src.dependencies import get_parsers


async def run_parsers() -> None:
    await asyncio.gather(
        *[p.parse() for p in get_parsers()]
    )
