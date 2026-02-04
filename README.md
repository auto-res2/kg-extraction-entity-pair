# KG Extraction from JacRED: Entity-Pair Guided Sequential Extraction

## 1. Background

### JacRED Dataset: Japanese Document-level Relation Extraction Dataset

- **Source**: https://github.com/YoumiMa/JacRED (clone to `/tmp/JacRED`)
- **Splits**: train (1400 docs), dev (300 docs), test (300 docs)
- **Format**: Each document has:
  - `title`: document title
  - `sents`: tokenized sentences (list of list of tokens)
  - `vertexSet`: entities with mentions (list of entity groups, each containing mention dicts with `name`, `type`, `sent_id`, `pos`)
  - `labels`: relations as `{h, t, r, evidence}` where h/t are vertexSet indices and r is a Wikidata P-code
- **9 entity types**: PER, ORG, LOC, ART, DAT, TIM, MON, %, NA
- **35 relation types**: Wikidata P-codes (P131, P27, P569, P570, P19, P20, P40, P3373, P26, P1344, P463, P361, P6, P127, P112, P108, P137, P69, P166, P170, P175, P123, P1441, P400, P36, P1376, P276, P937, P155, P156, P710, P527, P1830, P121, P674)
- **Statistics**: Avg ~17 entities/doc, avg ~20 relations/doc, avg ~253 chars/doc

## 2. Base Implementation (already provided)

The following files implement the baseline and two-stage extraction:

- **run_experiment.py**: Main orchestrator. Loads data, runs conditions (Baseline, Two-Stage), prints comparison table, saves results.json.
- **data_loader.py**: Data loading from JacRED JSON files, document selection (10 stratified from dev split), few-shot example selection, domain/range constraint table construction from training data.
- **llm_client.py**: Gemini API wrapper using `google-genai` library with Structured Outputs (`response_mime_type="application/json"` + `response_schema`), ThinkingConfig, and retry logic.
- **prompts.py**: All prompt templates including system prompt with 35 relation types defined in Japanese, extraction prompt (baseline and recall-oriented modes), and verification prompt for Stage 2.
- **extraction.py**: Two conditions:
  - `run_baseline()`: Single LLM call extraction with post-filtering (invalid labels, invalid entity types).
  - `run_proposed()`: Two-Stage generate+verify. Stage 1 extracts with recall-oriented prompt, Stage 2 batch-verifies candidates, then applies domain/range constraints.
- **evaluation.py**: Entity alignment (3-pass: exact match -> normalized match -> substring match) and micro-averaged P/R/F1 computation.
- **schemas.py**: JSON schemas for Gemini Structured Outputs (extraction schema with entities+relations, verification schema with decisions).

### Key code details

- Entity alignment maps predicted entity IDs to gold vertexSet indices using 3-pass matching.
- Domain/range constraints are built from training data: for each relation P-code, store the set of (head_type, tail_type) pairs observed.
- Verification (Stage 2) processes candidates in batches of 10, asking the LLM to judge each candidate.

## 3. Baseline Results (for comparison)

```
Model: gemini-3-flash-preview (thinking_budget=0)
              Precision   Recall     F1    TP    FP    FN
Baseline           0.26     0.16   0.20    24    70   124
Two-Stage          0.36     0.22   0.27    32    56   116
```

**Key issue**: Recall is very low (0.16-0.22). Most of the 148 gold relations are missed. The baseline approach extracts entities and relations together in one shot, which may cause the LLM to miss many valid entity pairs.

## 4. Environment Setup

```bash
# Clone JacRED dataset
git clone https://github.com/YoumiMa/JacRED /tmp/JacRED

# Install dependencies
pip install google-genai openai

# Set API key
export GEMINI_API_KEY="<your-key>"
```

## 5. API Configuration

- **Model**: `gemini-3-flash-preview` (recommended) or `gemini-2.0-flash`
- **Structured Outputs**: `response_mime_type="application/json"` + `response_schema` dict
- **Temperature**: 0.2
- **ThinkingConfig**: `thinking_budget=0` for speed, `2048` for quality
- Configuration is in `llm_client.py` (the `MODEL` constant and `call_gemini()` function)

## 6. Task: Implement Entity-Pair Guided Sequential Extraction

### Goal

Extract entities first, then systematically classify relations for each entity pair individually, ensuring no pair is missed.

### Design

1. **Step 1: Entity-only extraction** (1 LLM call)
   - Use a simplified schema that only asks for entities (no relations)
   - This focuses the LLM's attention purely on entity recognition

2. **Step 2: Pre-filter entity pairs using domain/range constraint table**
   - For each possible (head, tail) entity pair, check if their (head_type, tail_type) combination is valid for ANY relation type in the constraint table
   - Skip pairs whose entity types never co-occur for any relation in training data
   - This dramatically reduces the number of pairs to classify

3. **Step 3: Batch relation classification** (multiple LLM calls)
   - Group remaining pairs into batches of 20-30 pairs per LLM call
   - For each pair, ask: "What relation exists between E1 and E2? Choose from [applicable P-codes based on type pair] or NA"
   - The applicable P-codes are filtered: for a (PER, LOC) pair, only show relation types where (PER, LOC) appears in the constraint table
   - Include the full document text in each batch call for context

4. **Step 4: Collect results**
   - Gather all non-NA predictions as final triples
   - No additional verification needed (the per-pair classification is already precise)

### Implementation Details

- **Add `ENTITY_ONLY_SCHEMA` in `schemas.py`**:
  ```python
  ENTITY_ONLY_SCHEMA = {
      "type": "object",
      "properties": {
          "entities": {
              "type": "array",
              "items": {
                  "type": "object",
                  "properties": {
                      "id": {"type": "string"},
                      "name": {"type": "string"},
                      "type": {"type": "string", "enum": ["PER", "ORG", "LOC", "ART", "DAT", "TIM", "MON", "%"]},
                  },
                  "required": ["id", "name", "type"],
              },
          },
      },
      "required": ["entities"],
  }
  ```

- **Add `PAIR_CLASSIFICATION_SCHEMA` in `schemas.py`**:
  ```python
  PAIR_CLASSIFICATION_SCHEMA = {
      "type": "object",
      "properties": {
          "pair_decisions": {
              "type": "array",
              "items": {
                  "type": "object",
                  "properties": {
                      "pair_index": {"type": "integer"},
                      "relation": {"type": "string"},  # P-code or "NA"
                      "evidence": {"type": "string"},
                  },
                  "required": ["pair_index", "relation", "evidence"],
              },
          },
      },
      "required": ["pair_decisions"],
  }
  ```

- **Add `run_entity_pair()` function in `extraction.py`**:
  - Step 1: Call `call_gemini()` with entity-only prompt and `ENTITY_ONLY_SCHEMA`
  - Step 2: Generate all (head, tail) pairs, filter by constraint table
  - Step 3: For each batch of 20-30 filtered pairs, call `call_gemini()` with pair classification prompt
  - Step 4: Collect non-NA results, create Triple objects
  - Return entities, triples, and stats dict with counts

- **Add `build_entity_only_prompt()` in `prompts.py`**:
  - System prompt focusing on entity extraction only
  - Include few-shot example (entities only, no relations)

- **Add `build_pair_classification_prompt()` in `prompts.py`**:
  - Takes document text, a batch of (head_entity, tail_entity) pairs, and applicable relation types per pair
  - Asks the LLM to classify each pair

- **Pre-filter logic**:
  - Build a reverse lookup from the constraint table: `valid_type_pairs = set()` containing all (head_type, tail_type) tuples that appear for any relation
  - For each entity pair (i, j) where i != j, check if (entities[i].type, entities[j].type) is in valid_type_pairs
  - Also build per-pair applicable relations: for pair (PER, ORG), only P108, P463, P112, etc.

- **Update `run_experiment.py`** to add the third condition

### Expected Improvement

- **High Recall**: Every valid entity pair is explicitly checked, so relations are less likely to be missed.
- **Precision**: Per-pair classification with only applicable relation types should be precise.
- **Cost**: 1 entity call + ceil(filtered_pairs / 20) classification calls per document. With ~17 entities/doc, there are ~272 ordered pairs; after type filtering, estimate ~50-100 valid pairs, requiring 3-5 classification calls. Total: ~5-15x baseline cost.

### Evaluation

- Same P/R/F1 computation on the same 10 dev documents
- Report per-document results and aggregate metrics
- Also report: number of entity pairs generated, number after filtering, number of LLM calls
- Compare: Baseline vs Two-Stage vs Entity-Pair

### Output Format

The final comparison table should look like:
```
              Precision   Recall     F1    TP    FP    FN
Baseline           ...      ...    ...   ...   ...   ...
Two-Stage          ...      ...    ...   ...   ...   ...
EntityPair         ...      ...    ...   ...   ...   ...
```
