# Supporting the OpenAI Responses API in Delphyne alongside the current Chat Completions API

## Motivation
Delphyne currently supports only OpenAI Chat Completions API. Although it is the standard in the industry, the currently superior choice for OpenAI models is the new Responses API for a few reasons: 

    1. Most importantly, in Chat Completions, reasoning items are discarded between requests, making it impossible to benefit from cache utilization when using reasoning models. In Responses API, the reasoning state of the model can be persisted across multiple tool calls that follow the same user message. (Though not across multiple conversation turns between user and assistant)  
    2. Newer reasoning models like `gpt-5-pro` is only supported in Responses.
    3. Responses API adds new built-in tools like File Search, Code Interpreter, Image Generation, which might be used in Delphyne in future?

## Rationale
The old Chat Completions API should still be supported due to it being the standard for gemini, mistral, deepseek,... AI models. So, the obvious choice is to keep the old API and add the support for the new one. 

Alternatives:
- Creating a new `OpenAIResponsesModel` class similar to `OpenAICompatibleModel` and adding new helper functions to accomodate for the format changes
- Parametrizing `OpenAICompatibleModel` and helper functions with api_type. 

Here I think, the former is preferable, because otherwise unneccessary complexity is created.

## The Responses API
There are some changes in the sent dictionary formats.

```
context = [
    { "role": "system|user|assistant", "content": "What is the capital of France?" },    # EasyInputMessageParam
    ...,
    { "type": "reasoning", "encrypted_content": "5v9wn75n7v5p98w75", ...},               # ResponseReasoningItemParam
    ...,
    { "type": "function_call", "name": ..., "arguments": json..., "call_id": ... },      # ResponseFunctionToolCallParam
    ...,
    { "type": "function_call_output", "output":content ..., "call_id": ... },            # FunctionCallOutput
]

res = client.responses.create(
    model="gpt-5",
    store=False,                                                                # to use the API stateless

    tools=[{type, name, description, parameters(JSON schema), strict}],         # FunctionToolParam

    include=["reasoning.encrypted_content", "message.output_text.logprobs"],    # newly added 

    text={"format": {}, "verbosity":},                                          # replaces response_format field            
                                                                                # ResponseFormatTextJSONSchemaConfigParam (name, 
                                                                                # schema, type, description, strict)
    
    instructions=system_message,                                                # not neccessary
    input=context,     
    reasoning={"effort": ..., "summary": ...}                                                  
)
```

## Specification

### models.py
- new fields to `RequestOptions`: 
`verbosity: Literal["low", "medium", "high"]`, 
`reasoning_summary: Literal["auto", "concise", "detailed"]`   ->not neccessary right now
- also background option? which lets model run in background and one can poll the status

### standard_models.py
- new `type OpenAIResponsesModelName = OpenAIModelName | ...` : can also include models `gpt-5-pro`.
- new function `openai_responses_model(model: OpenAIResponsesModelName, ...) -> OpenAIResponsesModel` similar to existing `openai_model()`
- new `_openai_responses_model(...) -> OpenAIResponsesModel` factory function or modification to `_openai_compatible_model(...)`
- modify `standard_model() -> StandardModel` to have an `api_type: Literal["chat_completion", "responses"]` parameter
    - if `api_type` is `responses` but non-openai model selected -> `ValueError`

#### New
- StandardModel(superclass of OpenAICompatible and OpenAIResponses) has parameter send_reasoning_tokens
- for standard_model() -> use just responses for openai
- ResponsesModel global option send_reasoning_tokens
- good error messages -> defensove programming
- openai_model(resend_reasoning, responses_api) -> default responses, default resend 

### openai_api.py
- new `translate_chat_responses_api(..) -> Sequence[ResponseInputItemParam]`
    - In Responses API the `input` format is as seen above
    - Treating system messages separately in the new `instructions` argument is not really neccessary -> leave `case SystemMessage` and `case UserMessage` as is!
    - flatten `answer.tool_calls` conversion, add each tool call as a separate item to the output list (see new format above)
    - flatten `ToolMessage` conversion (see new format above) 
    
- new class `OpenAIResponsesModel` similar to `OpenAICompatibleModel`. In `_send_final_request`:
    - change api access
    response: openai.responses.Response = create(
        model                             -> no change needed (some new model like gpt-5-pro only supported in responses)
        input                             -> instead of messages (should use the new `translate_chat_responses_api()`)
        num_completions                   -> not supported
        temperature                       -> keep
        reasoning: {"effort": .., "summary": ..}                   -> instead of reasoning_effort
        max_output_tokens                                          -> instead of max_completions_tokens 
        logprobs                                                   -> moved to include
        top_logprobs                                               -> keep
        tools                                                      -> new format (see above)
        text: {"format": {..}, "verbosity": ..}                    -> instead of response_format
        tool_choice                                                -> keep
        
    - add following options
        include = ["reasoning.encrypted_content", optionally "message.output_text.logprobs"]
        store = False
    )
    - Output handling: `response.output` is of type `list[ResponseFunctionToolCall | ResponseReasoningItem | ResponseOutputMessage]`. 
        Iterate over the list:
        - match over the type and fill the fields of `LLMOutput` in the cases of tool call and output message 
        - e.g. `response.output[0].content[0].text` (or .refusal) is the message, e.g. `response.output[1].arguments` (or .call_id or .name) for tool calls
    - to determine `FinishReason` look at response.status (`completed`, `failed`, `in_progress`, `cancelled`, `queued`, or
    `incomplete`). If incomplete, then look at `response.incomplete_details.reason` (`max_output_tokens` or `content_filter`). `tool_calls` is not a finish reason anymore.
    - To simulate the now missing `num_completions` option -> send multiple requests sequentially and collect `LLMOutput`s to a single `LLMResponse`? Or use AsyncAPI ?
    - new `stream_request` -> same but use chunck.delta? (ResponseStreamEvent is union of many delta event types. Check for ResponseTextDeltaEvent)
    
- new helper similar to `_make_chat_tool`-> flattened format (see above)
- new helper similar to `_chat_response_format` -> flattened format (see above)
- new helper similar to `_compute_spent_budget_responses` -> use `ResponseUsage` (input_tokens: int, input_tokens_details: InputTokensDetails (cached tokens)
    output_tokens: int, output_tokens_details: OutputTokensDetails (reasoning tokens), total_tokens: int) 
    instead of old `CompletionUsage` -> different field names, otherwise identical
- new helper similar to `translate_logprob_info` -> small naming changes, `response.output[0].content[0].text.logprobs` is a `list[Logprob]`

    
## Testing

- Mirror current tests for the OpenAI API for the new API
- Test strategy runs with the new API


## Num completions
- Research and send an email