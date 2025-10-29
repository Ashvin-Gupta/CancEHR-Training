# Refactoring Summary

## Overview

The Nightingale repository has been refactored to organize code by **data format** (tokens/text/embedded) rather than by experiment type. This creates three self-contained pipelines that are easier to understand, maintain, and extend.

## What Was Done

### Phase 1: New Structure Created

✅ Created new directory structure:
```
src/pipelines/
├── token_based/
│   ├── models/
│   ├── configs/
│   └── pretrain.py
├── text_based/
│   ├── configs/
│   ├── pretrain_llm.py
│   ├── finetune_bert.py
│   └── finetune_llm.py
├── embedded_based/
│   ├── models/
│   ├── configs/
│   ├── create_embeddings.py
│   ├── create_vocab_embeddings.py
│   ├── pretrain.py
│   └── finetune.py
└── shared/
    ├── base_models.py
    ├── blocks/
    └── note_models/
```

✅ Created `__init__.py` files for all new packages

### Phase 2: Token-Based Pipeline

✅ Moved token-based models:
- `transformer_decoder.py` → `src/pipelines/token_based/models/`
- `lstm.py` → `src/pipelines/token_based/models/`
- `gpt2.py` → `src/pipelines/token_based/models/`

✅ Updated imports in all token-based models to use new paths

✅ Removed `@register_model` decorators (no longer using registry)

✅ Created new training script:
- `src/pipelines/token_based/pretrain.py` (adapted from `src/experiments/run.py`)
- Supports LSTM, Transformer Decoder, GPT-2, LSTM with notes

✅ Migrated configs:
- `encoder_lstm.yaml`
- `cprd_decoder_lstm_test.yaml`
- `archive/*` configs

### Phase 3: Text-Based Pipeline

✅ Copied text-based scripts:
- `run_llm_pretrain.py` → `pretrain_llm.py`
- `run_hf_finetune.py` → `finetune_bert.py`
- `run_llm.py` → `finetune_llm.py`

✅ Migrated configs:
- `llm_pretrain.yaml`
- `fine-tune-bert.yaml`
- `fine-tune-bert2.yaml`
- `fine-tune-cpu.yaml`
- `sweep.yaml`

### Phase 4: Embedded-Based Pipeline

✅ Moved embedded models:
- `transformer_encoder_embedded.py` → `src/pipelines/embedded_based/models/`
- `transformer_decoder_embedded.py` → `src/pipelines/embedded_based/models/`

✅ Updated imports in embedded models

✅ Removed `@register_model` decorators

✅ Copied embedded scripts:
- `create_embedding_corpus.py` → `create_embeddings.py`
- `create_vocabulary_embeddings.py` → `create_vocab_embeddings.py`
- `pretrain_embedded.py` → `pretrain.py`
- `finetune_embedded.py` → `finetune.py`

✅ Updated imports in embedded scripts to use new model paths

✅ Migrated configs:
- `embed_text.yaml` → `create_embeddings.yaml`
- `pretrain_decoder_embedded.yaml`
- `finetune_decoder_embedded.yaml`
- `finetune_encoder_embedded.yaml`

### Phase 5: Shared Components

✅ Moved shared components:
- `src/models/base.py` → `src/pipelines/shared/base_models.py`
- `src/models/blocks/` → `src/pipelines/shared/blocks/`
- `src/models/note_models/` → `src/pipelines/shared/note_models/`

### Phase 6: Training Module

✅ Created `src/training/token_trainer.py` (copy of `train.py`)
- Clarifies it's for token-based training

### Phase 7: Entry Point Scripts

✅ Created convenience shell scripts at project root:
- `run_token_pretrain.sh`
- `run_text_pretrain.sh`
- `run_text_finetune_bert.sh`
- `run_embedded_pipeline.sh`

✅ Made all scripts executable

### Phase 8: Documentation

✅ Updated main `README.md`:
- New architecture overview
- Quick start for all three pipelines
- Pipeline comparison table
- Migration guide

✅ Created pipeline-specific READMEs:
- `src/pipelines/token_based/README.md` (comprehensive guide)
- `src/pipelines/text_based/README.md` (comprehensive guide)
- `src/pipelines/embedded_based/README.md` (comprehensive guide)

✅ Created deprecation notice:
- `src/experiments/_DEPRECATED_README.md` (full migration guide)

## File Changes Summary

### New Files Created (47 total)

**Directory Structure:**
- 7 `__init__.py` files for new packages

**Models:**
- 3 token-based models (copied with updated imports)
- 2 embedded-based models (copied with updated imports)
- 3 shared component directories

**Scripts:**
- 1 token-based training script
- 3 text-based scripts (copied)
- 4 embedded-based scripts (copied with updated imports)
- 4 convenience shell scripts

**Configs:**
- 3 token-based configs + archive
- 5 text-based configs
- 4 embedded-based configs

**Documentation:**
- 1 main README (updated)
- 3 pipeline READMEs
- 1 deprecation notice
- 1 refactoring summary (this file)

### Files Preserved

The following directories remain unchanged:
- `src/data/` - All dataset loaders work with new structure
- `src/training/` - Training utilities (added token_trainer.py copy)
- `src/evaluation/` - Evaluation tools
- `src/experiments/` - Marked as deprecated, files kept for reference

## Breaking Changes

### Import Changes Required

**Before:**
```python
from src.models.core_models.lstm import LSTM
from src.models.core_models.transformer_encoder_embedded import TransformerEncoderEmbedded
from src.models.base import BaseNightingaleModel
```

**After:**
```python
from src.pipelines.token_based.models.lstm import LSTM
from src.pipelines.embedded_based.models.transformer_encoder_embedded import TransformerEncoderEmbedded
from src.pipelines.shared.base_models import BaseNightingaleModel
```

### Entry Point Changes

**Before:**
```bash
python -m src.experiments.run --config_name encoder_lstm --experiment_name exp
```

**After:**
```bash
python -m src.pipelines.token_based.pretrain --config path/to/config.yaml --experiment_name exp
# Or use convenience script:
./run_token_pretrain.sh path/to/config.yaml exp
```

### Config Path Changes

All configs moved from `src/experiments/configs/` to pipeline-specific `configs/` directories.

### Model Registry Removed

The `@register_model` decorator and registry system are no longer used. Models are imported directly.

## Testing Checklist

Before merging, test each pipeline:

### Token-Based Pipeline
```bash
# Test with a small config
./run_token_pretrain.sh src/pipelines/token_based/configs/encoder_lstm.yaml test_exp

# Verify:
# - Script runs without import errors
# - Model loads correctly
# - Training completes (at least 1 epoch)
# - Results saved to results/token_based/test_exp/
```

### Text-Based Pipeline
```bash
# Test BERT fine-tuning
./run_text_finetune_bert.sh src/pipelines/text_based/configs/fine-tune-bert.yaml

# Test LLM pretraining
./run_text_pretrain.sh src/pipelines/text_based/configs/llm_pretrain.yaml

# Verify:
# - Scripts run without import errors
# - Models load correctly
# - Training starts
# - Results saved correctly
```

### Embedded Pipeline
```bash
# Test full pipeline
./run_embedded_pipeline.sh \
    src/pipelines/embedded_based/configs/create_embeddings.yaml \
    src/pipelines/embedded_based/configs/pretrain_decoder_embedded.yaml \
    src/pipelines/embedded_based/configs/finetune_encoder_embedded.yaml

# Verify:
# - Embeddings created successfully
# - Models load with correct imports
# - Training completes
# - Results saved correctly
```

### Evaluation
```bash
# Test visualization server
python -m src.evaluation.visualisation_server.main

# Verify:
# - Server starts
# - Can load models from new paths
# - Visualizations work
```

## What Still Needs to be Done

### Updates to Config Files (User Action Required)

All copied configs need path updates:

**Token-based configs:**
- Update `data.train_dataset_dir` paths
- Update `data.vocab_path` paths
- Verify all paths are absolute or relative to project root

**Text-based configs:**
- Update `data.data_dir` paths
- Update lookup file paths
- Update `training.output_dir` to `results/text_based/...`

**Embedded configs:**
- Update `data.data_dir` paths
- Update `data.embedding_output_dir` to `embeddings/...`
- Update `training.output_dir` to `results/embedded_based/...`

**Sweep config:**
- Update `program` path to new script location

### Import Updates in Other Files

Files that may need import updates:
- `src/evaluation/` scripts if they import models
- Any notebooks in the repository
- Any utility scripts that import models

Search for these patterns and update:
```bash
# Find old imports
grep -r "from src.models.core_models" src/
grep -r "from src.models.base" src/
grep -r "from src.models.blocks" src/
```

### Registry Cleanup

Consider removing or updating:
- `src/models/registry.py` (no longer used)
- `src/models/utils.py` (load_model function not used by new pipelines)
- `src/models/__init__.py` (may need updating or removal)

## Benefits of New Structure

✅ **Clear organization**: Each pipeline is self-contained with models, configs, and scripts together

✅ **Easier navigation**: Know exactly where to look for code based on data format

✅ **Better onboarding**: New users can understand the three distinct approaches quickly

✅ **Independent testing**: Each pipeline can be tested and developed independently

✅ **Cleaner documentation**: Each pipeline has comprehensive documentation

✅ **Maintainability**: Changes to one pipeline don't affect others

✅ **Extensibility**: Easy to add new models to the appropriate pipeline

## Git Branch

All changes are on branch: `refactor/organize-by-data-format`

To continue development:
```bash
git checkout refactor/organize-by-data-format
# Make any needed updates
git add .
git commit -m "Update configs and test pipelines"
```

## Next Steps

1. **Update config files** with correct paths for your environment
2. **Test each pipeline** with real data (small experiments)
3. **Update any custom scripts** that import from old paths
4. **Update evaluation scripts** if needed
5. **Run integration tests** (training + evaluation)
6. **Review and merge** the refactoring branch

## Questions or Issues?

Refer to:
- Main README: `README.md`
- Pipeline READMEs: `src/pipelines/{pipeline}/README.md`
- Migration guide: `src/experiments/_DEPRECATED_README.md`
- This summary: `REFACTORING_SUMMARY.md`

