# utils/sync_wrapper.py
import asyncio
from .query_processor import QueryProcessor

class SyncQueryProcessor:
    def __init__(self):
        self.async_processor = QueryProcessor()
        # Initialize both models
        asyncio.run(self.async_processor.initialize())
    
    def process_query(self, user, query):
        """Synchronous interface that properly handles the async call"""
        async def _process():
            return await self.async_processor.process_query_async(user, query)
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_process())
            return result
        finally:
            loop.close()