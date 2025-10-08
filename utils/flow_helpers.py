import functools
import logging

def handle_exceptions(_func=None, *, default_return=None):
    """
    Decorator to handle exceptions in flow step methods.
    
    Can be used with or without parentheses:
    - Without params: @handle_exceptions
    - With params: @handle_exceptions(default_return={...})
    
    When an exception occurs, it logs the error and returns the default_return dict if provided,
    otherwise returns a generic failure dict.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"Exception in {func.__name__}: {e}")
                return default_return or {
                    "success": False,
                    "message": f"Step '{func.__name__}' failed: {str(e)}",
                }
        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)

