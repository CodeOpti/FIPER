"""Prompt for synthesizing executable I/O tests."""

prompt_dict = {
    "system_prompt_Python": (
        "You are a senior software test engineer. Generate executable, diverse "
        "input/output tests for Python programs. Cover normal, boundary, and "
        "error-prone cases."
    ),
    "system_prompt_Cpp": (
        "You are a senior software test engineer. Generate executable, diverse "
        "input/output tests for C++ programs. Cover normal, boundary, and "
        "error-prone cases."
    ),
    "instruction_Python": (
        "Reference examples:\n{four_example_prompt}\n\n"
        "Program under test:\n```python\n{Slow_program}\n```\n\n"
        "Generate at least 50 independent test inputs. Do not infer or reproduce "
        "benchmark public, private, or hidden tests. Outputs are intentionally omitted "
        "because the original slow program will label each input as the behavioral oracle. "
        "Return exactly one JSON object with this schema:\n"
        '{{"inputs": ["first complete stdin string", "second complete stdin string"]}}'
    ),
    "instruction_Cpp": (
        "Reference examples:\n{four_example_prompt}\n\n"
        "Program under test:\n```cpp\n{Slow_program}\n```\n\n"
        "Generate at least 50 independent test inputs. Do not infer or reproduce "
        "benchmark public, private, or hidden tests. Outputs are intentionally omitted "
        "because the original slow program will label each input as the behavioral oracle. "
        "Return exactly one JSON object with this schema:\n"
        '{{"inputs": ["first complete stdin string", "second complete stdin string"]}}'
    ),
    "IOtext": "## Test Case {io_id}:\nInput: {IOinput}\nOutput: {IOoutput}\n\n",
}
