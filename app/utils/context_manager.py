from typing import List, Callable, Any
from app.utils.logger import logger

class ContextManager:
    """
    A utility class to manage and scale context for LLMs.
    Provides Map-Reduce and Rolling-Window strategies to summarize or process 
    large arrays of data without exceeding context windows.
    """

    def __init__(self):
        pass

    def map_reduce(self, items: List[str], batch_size: int, reduce_func: Callable[[List[str]], str]) -> str:
        """
        Recursively processes a list of items using a Map-Reduce approach.
        Groups items into batches, reduces each batch using `reduce_func`, 
        and repeats until only one item remains.
        
        Args:
            items: List of text strings (e.g., chunk summaries).
            batch_size: Number of items per batch.
            reduce_func: Function that takes a list of strings and returns a single summarized string.
            
        Returns:
            A single string representing the final reduced context.
        """
        if not items:
            return ""
        
        if len(items) == 1:
            return items[0]

        logger.info(f"ContextManager: Map-Reducing {len(items)} items with batch_size={batch_size}")
        
        next_level_items = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            logger.info(f"ContextManager: Reducing batch of {len(batch)} items...")
            reduced_item = reduce_func(batch)
            next_level_items.append(reduced_item)
            
        # Recursive call if we still have more than 1 item
        return self.map_reduce(next_level_items, batch_size, reduce_func)

    def rolling_summarize(self, items: List[str], update_func: Callable[[str, str], str], initial_state: str = "") -> str:
        """
        Processes items sequentially, updating a rolling state.
        Best for maintaining strict chronological narrative, but slower than Map-Reduce.
        
        Args:
            items: List of text strings.
            update_func: Function that takes (current_state, new_item) and returns updated_state.
            initial_state: Starting state string.
            
        Returns:
            The final accumulated state.
        """
        current_state = initial_state
        for i, item in enumerate(items):
            logger.info(f"ContextManager: Rolling summarize item {i+1}/{len(items)}...")
            current_state = update_func(current_state, item)
            
        return current_state
