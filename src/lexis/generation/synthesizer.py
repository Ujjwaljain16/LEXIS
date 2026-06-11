"""
True SSE Streaming Synthesizer for LEXIS.

Rationale: Generates the final answer using Groq while yielding tokens instantly to hit TTFT < 1s.
Source Inspiration: plan.md (Section 5) and native FastAPI StreamingResponses.
Deviations from Source Repos: We use AsyncGroq with stream=True. We also embed citations directly into the prompt instructing the LLM to cite by index.
Expected Impact on Metrics: Radically improves perceived latency.
"""
import re
from litellm import acompletion
from typing import AsyncGenerator, List, Dict, Any, Set
from lexis.config import settings

class LexisSynthesizer:
    def __init__(self):
        pass

    def _build_context_string(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats the packed chunks into a numbered context block for citation."""
        context_str = ""
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text") or chunk.get("content") or chunk.get("proposition") or chunk.get("questions")
            if isinstance(text, list):
                text = " ".join(text)
            source_path = chunk.get("_source_path", "UNKNOWN")
            context_str += f"[{i}] (Source: {source_path}): {text}\n\n"
        return context_str

    async def stream_answer(self, query: str, packed_chunks: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        context_str = self._build_context_string(packed_chunks)
        
        system_prompt = (
            "You are LEXIS, an elite legal and analytical reasoning AI. "
            "Answer the user's question using ONLY the provided context. "
            "You MUST cite the context using the provided bracketed indices, e.g., [1], [2]. "
            "If the context does not contain the answer, state explicitly that you do not have enough information."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

        valid_keys: Set[str] = {c.get("chunk_id") for c in packed_chunks if c.get("chunk_id")}
        
        try:
            stream = await acompletion(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                stream=True
            )
            
            # Sliding window buffer to intercept pqac- keys
            buffer = ""
            pqac_regex = re.compile(r"pqac-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}")
            
            async for chunk in stream:
                token = chunk.choices[0].delta.content
                if not token:
                    continue
                    
                buffer += token
                
                # If 'pqac-' is in the buffer, hold yielding until we have the full UUID (41 chars)
                if "pqac-" in buffer:
                    idx = buffer.find("pqac-")
                    # If we don't have enough characters to evaluate the regex, keep buffering
                    if len(buffer) - idx < 41:
                        continue
                        
                    # We have enough characters to regex match
                    match = pqac_regex.search(buffer)
                    if match:
                        found_key = match.group(0)
                        if found_key in valid_keys:
                            # Valid citation, yield it
                            yield buffer[:match.end()]
                            buffer = buffer[match.end():]
                        else:
                            # Hallucinated key, drop it from the buffer entirely!
                            buffer = buffer[:match.start()] + buffer[match.end():]
                    else:
                        # False alarm or malformed pqac-, just flush
                        yield buffer
                        buffer = ""
                else:
                    # No citation marker, yield immediately
                    yield buffer
                    buffer = ""
            
            # Flush remaining
            if buffer:
                yield buffer
                
        except Exception as e:
            yield f"\n\n[Error during synthesis: {str(e)}]"
