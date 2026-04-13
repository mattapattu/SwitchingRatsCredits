# DSL: Dynamic Structure Learning

Implementation of the **Dynamic Structure Learning (DSL)** framework, as described in:

> **Inferring Time-Varying Internal Models of Agents Through Dynamic Structure Learning**  
> Ashwin Moongathottathil James, Ingrid Bethus, Alexandre Muzy  
> *Behavioral Neuroscience, Vol. 140, No. 2, 75–89 (2026)*  
> DOI: [10.1037/bne0000642](https://doi.org/10.1037/bne0000642)

---

## Overview

DSL is a framework for inferring how an agent's internal structure — defined as a combination
of a **learning rule** and an **environment representation** — evolves over time from observed behavior.

Applied here to rat behavior in a T-maze task, DSL recovers whether each rat used:
- **CoACA** (Cognitive Activity-Based Credit Assignment) — a heuristic rule
- **Q-learning** — a value-based rational rule
- **MR-s** — a suboptimal maze representation (with state aliasing)
- **MR-o** — an optimal maze representation

Inference is performed using a **Particle Stochastic Approximation EM (PSAEM)** algorithm
with **Conditional Particle Filter with Ancestor Sampling (CPF-AS)**.

---

## Repository Structure

| Path | Description |
|------|-------------|
| `src/` | C++ source files (CPF-AS, EM, simulation, inference) |
| `include/` | Header files |
| `data/` | Rat behavioral trajectory data (.txt) |
| `results/` | Fitted model outputs (.Rdata) |
| `Makefile` | Build configuration |

---

## Dependencies

- C++17 compiler (g++ recommended)
- [Pagmo2](https://esa.github.io/pagmo2/) — for differential evolution optimization
- [Boost Graph Library](https://www.boost.org/doc/libs/release/libs/graph/) — for maze graph representation
- [BS::thread_pool](https://github.com/bshoshany/thread-pool) — for parallelism (included)

---

## Build & Run

```bash
# Build
make

# Run inference on rat data
./InferStrategy

# Run simulations
./GenerateSimulation
```

---

## Key Results

DSL correctly identified that:
- **Slow-learning rats** (rat1, rat2, rat3) began with CoACA-s (heuristic + suboptimal maze),
  then transitioned to QL-o (Q-learning + optimal maze)
- **Fast-learning rats** (rat4, rat5) used QL-o from the start
- Agent structure recovery rate exceeded **90%** on simulated data

---

## Citation

```bibtex
@article{james2026dsl,
  title={Inferring Time-Varying Internal Models of Agents Through Dynamic Structure Learning},
  author={Moongathottathil James, Ashwin and Bethus, Ingrid and Muzy, Alexandre},
  journal={Behavioral Neuroscience},
  volume={140},
  number={2},
  pages={75--89},
  year={2026},
  doi={10.1037/bne0000642}
}
```

---

## Note

This paper is published in *Behavioral Neuroscience* (APA). For access to the full paper,
see the DOI link above. For questions, contact: ashwin.moongathottathil.james@hu-berlin.de
