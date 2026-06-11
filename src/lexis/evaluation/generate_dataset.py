import os
import json
import asyncio
import logging
from pathlib import Path
from litellm import acompletion
import kagglehub
from lexis.ingestion.pipeline import IngestionPipeline
from lexis.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def extract_questions_from_chunk(chunk_id: str, content: str) -> list:
    prompt = f"""
    You are a legal expert evaluating a Retrieval-Augmented Generation (RAG) system.
    Based on the following legal contract excerpt, generate 1 highly specific, realistic legal question that can be answered EXACTLY by this text.
    
    CRITICAL INSTRUCTIONS:
    - Generate a semantically equivalent question.
    - DO NOT copy the exact wording or phrases from the source text.
    - Use natural legal language.
    - Do not ask broad questions. Ask about specific numbers, termination conditions, or specific obligations.
    
    Return ONLY a JSON array of objects, each containing:
    {{"query": "The question text"}}

    Excerpt:
    {content}
    """
    
    try:
        response = await acompletion(
            model=settings.gemini_model_synthesis,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            temperature=0.2,
            api_key=settings.gemini_api_key
        )
        res_json = json.loads(response.choices[0].message.content)
        # Handle cases where it returns a dict with a list inside
        if "questions" in res_json:
            return [q["query"] for q in res_json["questions"]]
        elif isinstance(res_json, list):
            return [q.get("query") for q in res_json]
        elif "query" in res_json:
            return [res_json["query"]]
    except Exception as e:
        logger.error(f"Failed to generate question for {chunk_id}: {e}")
        return []
    return []

async def generate_dataset():
    logger.info("Downloading Atticus Open Contract Dataset via kagglehub...")
    # NOTE: Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables
    dataset_path = kagglehub.dataset_download("theatticusproject/atticus-open-contract-dataset-aok-beta")
    logger.info(f"Dataset downloaded to {dataset_path}")
    
    # We will pick 10 PDF or TXT contracts from the dataset
    all_files = list(Path(dataset_path).rglob("*.pdf")) + list(Path(dataset_path).rglob("*.txt"))
    selected_files = all_files[:10]
    
    if len(selected_files) < 10:
        logger.warning(f"Only found {len(selected_files)} contracts. Proceeding anyway.")
    
    pipeline = IngestionPipeline()
    
    eval_questions = []
    total_questions_needed = 50
    
    # Process each contract
    for idx, file_path in enumerate(selected_files):
        doc_id = f"aok_contract_{idx}"
        logger.info(f"Ingesting {doc_id}: {file_path}")
        
        # 1. Ingest document to generate exactly the chunks that exist in the system
        await pipeline.ingest_document(str(file_path), doc_id)
        
        # 2. To generate questions, we should intercept the chunks. 
        # Since ingestion pipeline persists them to ES, we can fetch them.
        # However, for simplicity of this script, let's parse and chunk them manually here as well 
        # just to grab the Chunk objects, since we want to attach the EXACT UUID.
        elements = pipeline.parser.parse(str(file_path), doc_id)
        chunks = pipeline.chunker.chunk(elements)
        
        if not chunks:
            continue
            
        # Select up to 5 diverse chunks from this document to generate questions
        # We sample evenly across the document
        step = max(1, len(chunks) // 5)
        sampled_chunks = chunks[::step][:5]
        
        for chunk in sampled_chunks:
            if len(eval_questions) >= total_questions_needed:
                break
                
            queries = await extract_questions_from_chunk(chunk.chunk_id, chunk.content)
            for q in queries:
                eval_questions.append({
                    "query": q,
                    "expected_chunk_ids": [chunk.chunk_id],
                    "metadata": {
                        "contract": doc_id,
                        "category": chunk.metadata.document_type,
                        "difficulty": "medium"
                    }
                })
                
                if len(eval_questions) >= total_questions_needed:
                    break
        
        if len(eval_questions) >= total_questions_needed:
            break

    output_file = "evaluation/datasets/cuad_foundation_v1.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"questions": eval_questions}, f, indent=2)
        
    logger.info(f"Generated {len(eval_questions)} questions saved to {output_file}")
    logger.info("NOTE: Human verification of these questions is required before running the official benchmark.")

if __name__ == "__main__":
    asyncio.run(generate_dataset())
