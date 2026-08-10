"""Prompt for semantics-preserving performance optimization."""

prompt_dict = {
    "system_prompt_Python": (
        "You are an expert Python performance engineer. Preserve exact behavior "
        "while reducing runtime and memory usage. Return only valid Python code."
    ),
    "system_prompt_Cpp": (
        "You are an expert C++ performance engineer. Preserve exact behavior "
        "while reducing runtime and memory usage. Return only valid C++ code."
    ),
    "instruction_Python": (
        "Optimize the following Python program without changing its behavior.\n\n"
        "Slow program:\n```python\n{Slow_program}\n```\n\n"
        "Semantic overview:\n{overview_description}\n\n"
        "Validation I/O examples:\n{IO_examples}\n\n"
        "Return only the optimized program in a Python code block."
    ),
    "instruction_Cpp": (
        "Optimize the following C++ program without changing its behavior.\n\n"
        "Slow program:\n```cpp\n{Slow_program}\n```\n\n"
        "Semantic overview:\n{overview_description}\n\n"
        "Validation I/O examples:\n{IO_examples}\n\n"
        "Return only the optimized program in a C++ code block."
    ),
    "IOtext": "## Test Case {io_id}:\nInput: {IOinput}\nOutput: {IOoutput}\n\n",
}
