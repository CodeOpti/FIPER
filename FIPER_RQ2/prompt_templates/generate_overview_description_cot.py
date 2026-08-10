"""Prompt for generating a concise semantic overview."""

prompt_dict = {
    "system_prompt_Python": (
        "You are a software documentation expert. Describe the behavior of the "
        "Python program precisely enough to guide semantics-preserving optimization."
    ),
    "system_prompt_Cpp": (
        "You are a software documentation expert. Describe the behavior of the "
        "C++ program precisely enough to guide semantics-preserving optimization."
    ),
    "instruction_Python": (
        "Explain this Python program and its input/output behavior.\n\n"
        "```python\n{Slow_program}\n```\n\n"
        "Observed I/O examples:\n{IO_examples}\n\n"
        "Return one concise paragraph beginning with the program's main behavior."
    ),
    "instruction_Cpp": (
        "Explain this C++ program and its input/output behavior.\n\n"
        "```cpp\n{Slow_program}\n```\n\n"
        "Observed I/O examples:\n{IO_examples}\n\n"
        "Return one concise paragraph beginning with the program's main behavior."
    ),
    "IOtext": "## Test Case {io_id}:\nInput: {IOinput}\nOutput: {IOoutput}\n\n",
}
