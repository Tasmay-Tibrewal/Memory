# Development Philosophy

This document captures the design, implementation, architecture, coding, and documentation philosophy used in this project. It serves as a reference for future sessions and agents to maintain consistency.

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [Architecture Philosophy](#architecture-philosophy)
3. [Implementation Philosophy](#implementation-philosophy)
4. [Coding Style](#coding-style)
5. [Configuration Philosophy](#configuration-philosophy)
6. [Documentation Philosophy](#documentation-philosophy)
7. [Project Structure Philosophy](#project-structure-philosophy)
8. [Session Management Philosophy](#session-management-philosophy)

---

## Core Principles

### 1. Flexibility Over Rigidity
- **Prefer configuration flags** over hard-coded behavior
- **Enable ablation studies** through toggles (e.g., `vanilla_mode`)
- **Make everything configurable** but provide sensible defaults
- **Allow multiple approaches** (e.g., both variant A and B block structures)

### 2. Explicit Over Implicit
- **Document design decisions** with rationale, not just what but *why*
- **Name things descriptively** even if longer (e.g., `memory_layer_placement` not `mem_place`)
- **Avoid magic numbers** - define constants and explain them
- **Comments explain intent**, code explains what

### 3. Modularity Over Monoliths
- **Single responsibility** - each file does one thing well
- **Clear boundaries** - modules have well-defined interfaces
- **Minimal coupling** - modules depend on abstractions, not implementations
- **Easy to test in isolation** even without formal unit tests

### 4. Research-Friendly
- **Support experiments** through config flags
- **Enable comparisons** (vanilla mode, LoRA-only mode, etc.)
- **Document limitations honestly** in design.md
- **Provide recommended starting points** in example configs

---

## Architecture Philosophy

### Component Organization
```
Core logic (memory_transformer/) → Training (training/) → Scripts (scripts/)
                ↓
        Inference (inference/)
```

### Dependency Direction
- **Core has no external dependencies** to other packages
- **Higher-level modules depend on lower-level** (trainer depends on model, not reverse)
- **Avoid circular imports** through lazy loading where needed

### Layer Separation
```
config.py      ← Configuration (what to build)
memory_bank.py ← Data structures (memory storage)
memory_attention.py ← Computation (how to access memory)
memory_block.py ← Composition (blocks using attention)
model.py / adapter.py ← Integration (full models)
```

### Extension Points
- **Factory functions** for creating components (e.g., `create_memory_bank()`)
- **Class inheritance** for variants (e.g., `StandardMemoryBank`, `FactorizedMemoryBank`)
- **Configuration-driven** behavior selection

---

## Implementation Philosophy

### General Approach
1. **Read requirements thoroughly** before implementing
2. **Consult with user** on major architectural decisions
3. **Implement incrementally** - core first, then features
4. **Verify as you go** - test imports, check for gaps

### Error Handling
- **Fail early with clear messages** for invalid configurations
- **Validate inputs** at boundaries (config loading, forward pass)
- **Graceful fallbacks** where appropriate (e.g., Flash Attention → standard)

### Performance Considerations
- **Correctness first**, then optimize
- **Document performance trade-offs** (memory vs speed)
- **Provide options** for different resource constraints (low-rank, gradient checkpointing)

### Research Code vs Production Code
This is research code, so:
- **Readability over micro-optimization**
- **Flexibility over maximum performance**
- **Documentation over terseness**
- **But still maintain code quality** (no hacks, clean interfaces)

---

## Coding Style

### Python Style

```python
# File structure
"""
Module docstring explaining purpose.
"""

from typing import Optional, List, Dict
import torch
import torch.nn as nn

# Constants
DEFAULT_VALUE = 0.02

# Classes
class ComponentName(nn.Module):
    """
    One-line description.
    
    Longer explanation if needed.
    
    Args:
        param1: Description
        param2: Description
    """
    
    def __init__(
        self,
        param1: int,
        param2: Optional[str] = None,
    ):
        super().__init__()
        self.param1 = param1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x

# Functions
def function_name(
    arg1: torch.Tensor,
    arg2: int = 10,
) -> torch.Tensor:
    """
    Brief description.
    
    Args:
        arg1: Description
        arg2: Description
        
    Returns:
        Description
    """
    return arg1
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `MemoryCrossAttention` |
| Functions | snake_case | `create_memory_bank` |
| Variables | snake_case | `hidden_states` |
| Constants | UPPER_SNAKE | `DEFAULT_INIT_STD` |
| Config fields | snake_case | `num_memory_tokens` |
| Private | Leading underscore | `_internal_method` |

### Type Hints
- **Always use type hints** for function signatures
- **Use Optional** for nullable parameters
- **Use Union** sparingly, prefer type-specific functions

### Docstrings
- **All public classes and functions** must have docstrings
- **Args, Returns, Raises** sections for non-trivial functions
- **Keep one-liners for simple methods** (e.g., `"""Forward pass."""`)

---

## Configuration Philosophy

### Structure
```yaml
# Group by domain, not by data type
model:      # What to build
memory:     # Memory-specific options
training:   # How to train
```

### Defaults
- **Sensible defaults** - project should run with minimal config
- **Conservative defaults** - prefer stability over performance
- **Research defaults** - enable features for full experimentation

### Naming
- **Descriptive names** - `memory_layer_placement` not `mlp`
- **Consistent prefixes** - all memory options start with `memory_` or are in `memory:` section
- **Boolean flags** - use `use_` prefix (e.g., `use_chapters: true`)

### Validation
- **Validate early** - at config loading time
- **Provide helpful errors** - what's wrong and how to fix it

---

## Documentation Philosophy

### Core Principles

1. **Documentation is a first-class deliverable**, not an afterthought
2. **Multiple levels of detail** for different audiences
3. **Living documentation** - update as code changes
4. **Self-documenting code** doesn't mean no documentation

### Documentation Levels

| Level | Document | Audience | Updates |
|-------|----------|----------|---------|
| Quick start | README.md | New users | On feature changes |
| Code docs | Package READMEs | Developers | On module changes |
| Deep dive | architecture.md | Contributors | On design changes |
| Rationale | design.md | Maintainers | On decisions |
| Current state | context.md | Agents/Handoffs | Every session |
| History | session.md | Audit/Debug | During session |
| Summary | session_summary.md | Quick context | End of session |

### README Structure for Packages
```markdown
# Package Name

Brief description.

## Module Overview
[Files and their purposes]

## Detailed Documentation
[Each file explained]

## Usage Examples
[Code examples]

## Dependencies
[What this package needs]
```

### Writing Style

- **Concise but complete** - don't over-explain obvious things
- **Tables for structured data** - easier to scan than prose
- **Code examples** - show don't just tell
- **Links between docs** - cross-reference related information

### Markdown Formatting

```markdown
# Heading 1 (document title only)
## Heading 2 (major sections)
### Heading 3 (subsections)

**Bold** for emphasis
`code` for file names, functions, commands
```code blocks``` for longer code

| Tables | For | Structured | Data |
|--------|-----|------------|------|

- Bullet lists for unordered items
1. Numbered lists for ordered items
```

### When to Update Docs

| Event | Update |
|-------|--------|
| New feature | README, architecture.md, session.md |
| Bug fix | session.md (if significant) |
| Design decision | design.md, session.md |
| Session end | context.md, session_summary.md |
| Structure change | All relevant READMEs |

---

## Project Structure Philosophy

### Directory Organization

```
Root/
├── README.md              # Entry point, quick start
├── requirements.txt       # Dependencies with comments
│
├── core_package/          # Main implementation
│   ├── README.md          # Package documentation
│   ├── __init__.py        # Clean exports
│   └── modules.py         # One responsibility per file
│
├── supporting_packages/   # Training, inference, etc.
│   └── ...
│
├── scripts/               # User entry points
│   └── ...
│
├── configs/               # Configuration examples
│   ├── README.md          # Config format documentation
│   └── *.yaml             # Example configs
│
└── docs/                  # Deep documentation
    ├── README.md          # Doc overview
    ├── architecture.md    # Technical deep dive
    ├── design.md          # Decisions and rationale
    ├── context.md         # Current state summary
    └── meta_artifacts/    # Session history
```

### File Naming

| Type | Convention |
|------|------------|
| Python | `snake_case.py` |
| YAML | `descriptive_name.yaml` |
| Markdown | `lowercase_with_underscores.md` or `camelCase.md` for standard names |

### What Goes Where

| Content | Location |
|---------|----------|
| Quick start | Root README.md |
| API documentation | Package README.md |
| Technical architecture | docs/architecture.md |
| Why decisions were made | docs/design.md |
| Current project state | docs/context.md |
| Session history | docs/meta_artifacts/ |

---

## Session Management Philosophy

### Context Preservation

1. **Log everything significant** in session.md during work
2. **Summarize at end** in session_summary.md
3. **Update current state** in context.md
4. **Keep artifacts organized** in meta_artifacts/sessionN/

### Handoff Readiness

At any point, another agent or developer should be able to:
1. Read `context.md` for current state (5 min)
2. Read `session_summary.md` for history overview (5 min)
3. Dive into `session.md` for details if needed (30 min)
4. Find any decision rationale in `design.md`

### Session Structure

```markdown
# Session N

## Objective
What we're trying to accomplish

## Work Completed
What was done (with details)

## Decisions Made
Any choices with rationale

## Issues Encountered
Problems and resolutions

## Next Steps
What remains to be done
```

### Update Frequency

| Document | When to Update |
|----------|----------------|
| session.md | Throughout session, as work happens |
| context.md | End of major phases or session |
| session_summary.md | End of session |
| design.md | When decisions are made |

---

## Summary of Key Principles

1. **Flexibility through configuration** - don't hardcode what can be configured
2. **Explicit documentation** - document not just what but why
3. **Modularity** - single responsibility, clear boundaries
4. **Research-friendly** - enable experiments and comparisons
5. **Multiple documentation levels** - from quick start to deep dive
6. **Living documentation** - update as code evolves
7. **Session logging** - preserve context for handoffs
8. **Sensible defaults** - work out of the box with minimal config

---

*This philosophy was developed during Session 1 of the Memory-Augmented Transformer project (February 5, 2026).*
