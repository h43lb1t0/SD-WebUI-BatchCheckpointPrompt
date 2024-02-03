import inspect
from pprint import pprint
from typing import Any

class Logger():
    """
        Log class with different styled logs.
        debugging can be enabled/disabled for the whole instance
    """

    def __init__(self) -> None:
        self.debug = False

    def log_caller(self) -> None:
        caller_frame = inspect.currentframe().f_back.f_back  # type: ignore
        caller_function_name = caller_frame.f_code.co_name # type: ignore
        caller_self = caller_frame.f_locals.get('self', None) # type: ignore
        if caller_self is not None:
            caller_class_name = caller_self.__class__.__name__
            print(f"\tat: {caller_class_name}.{caller_function_name}\n")
        else:
            print(f"\tat: {caller_function_name}\n")


    def debug_log(self, msg: str, debug: bool = False) -> None:
        if self.debug or debug:
            print(f"\n\tDEBUG: {msg}")
            self.log_caller()

    def pretty_debug_log(self, msg: Any) -> None:
        if self.debug:
            print("\n\n\n")
            pprint(msg)
            self.log_caller()


    def log_info(self, msg: str) -> None:
        print(f"INFO: Batch-Checkpoint-Prompt: {msg}")

    def debug_print_attributes(self, obj: Any) -> None:
        print("Atributes: ")
        if self.debug:
            attributes = dir(obj)
            for attribute in attributes:
                if not attribute.startswith("__"):
                    value = getattr(obj, attribute)
                    if not callable(value):  # Exclude methods
                        try:
                            print(f"{attribute}:")
                            pprint(value)
                        except:
                            print(f"{attribute}: {value}\n")
            print(f"\n{type(obj)}\n")
            self.log_caller()