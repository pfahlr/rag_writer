# Tool Agent JSON Schema

The tool agent communicates exclusively using JSON in one of two forms:

## 1. Tool Invocation
```json
{
  "tool": "<tool_name>",
  "args": {
    /* arguments matching the tool's input_schema */
  }
}
```

## 2. Final Response
```json
{
  "final": "<answer text>"
}
```

Any other output is considered invalid.

## Example Transcript
```text
System: <prompt listing tools>
User: "What is 2 + 2?"
Assistant: {"tool": "calculator", "args": {"expression": "2+2"}}
User: {"result": 4}
Assistant: {"final": "2 + 2 = 4"}
```
