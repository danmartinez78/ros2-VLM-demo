"""
Temporal VLM task-distillation pipeline (CR2-8B -> CR2-2B).

Sub-modules
-----------
schema      : canonical temporal sample / dataset schema
teacher     : teacher-label generation (abstracted runtime)
validator   : validation and filtering stage
dataset     : deterministic split / export
training    : student training config and dry-run launcher
evaluation  : temporal-quality evaluation hooks
"""
