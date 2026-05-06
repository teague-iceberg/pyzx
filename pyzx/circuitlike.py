"""Subgraph hierarchy for annotating regions of ZX graphs.

Subgraph          — a vertex set referencing a parent graph
  CircuitLike     — adds labeled input/output edges (qubit wires)
    CliffordUnitary — a CircuitLike known to be a Clifford gate
    PauliBox        — a CircuitLike known to be a Pauli box

An edge may appear as both an input and an output (with different qubit
labels), e.g. when two circuit-like subgraphs sharing a wire are merged
via vertical composition.

Constraint on CircuitLike: each vertex has at most 2 incident edges
with the same qubit label (one "in", one "out" along the wire).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set, Dict

from pyzx.graph.base import BaseGraph, VT, ET


# ---------------------------------------------------------------------------
# Subgraph hierarchy
# ---------------------------------------------------------------------------

@dataclass
class Subgraph:
    """A set of vertices annotating a region of a parent graph."""
    vertices: Set[VT]
    g: BaseGraph

    def boundary_edges(self) -> Set[ET]:
        """All edges crossing the subgraph boundary."""
        vset = self.vertices
        result: Set[ET] = set()
        for v in vset:
            for e in self.g.incident_edges(v):
                s, t = self.g.edge_st(e)
                other = t if s == v else s
                if other not in vset:
                    result.add(e)
        return result

    def internal_edges(self) -> Set[ET]:
        """Edges with both endpoints inside the subgraph."""
        vset = self.vertices
        result: Set[ET] = set()
        for v in vset:
            for e in self.g.incident_edges(v):
                s, t = self.g.edge_st(e)
                if s in vset and t in vset:
                    result.add(e)
        return result


@dataclass
class CircuitLike(Subgraph):
    """A subgraph with labeled input/output edges.

    Attributes:
        input_edges: Qubit label → edge for inputs.
        output_edges: Qubit label → edge for outputs.
            An edge may appear in both (with different qubit labels).
        edge_labels: Edge → qubit label for internal edges.
            Non-IO boundary edges must NOT appear here.
    """
    input_edges: Dict[int, ET] = field(default_factory=dict)
    output_edges: Dict[int, ET] = field(default_factory=dict)
    edge_labels: Dict[ET, int] = field(default_factory=dict)

    def neutral_edges(self) -> Set[ET]:
        """Boundary edges that are neither input nor output."""
        io = set(self.input_edges.values()) | set(self.output_edges.values())
        return self.boundary_edges() - io

    def input_qubit_map(self) -> Dict[int, ET]:
        """Map qubit label → input edge."""
        return dict(self.input_edges)

    def output_qubit_map(self) -> Dict[int, ET]:
        """Map qubit label → output edge."""
        return dict(self.output_edges)

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: list[str] = []

        boundary = self.boundary_edges()
        internal = self.internal_edges()
        valid_io = boundary | internal  # IO edges may be boundary or internal

        # Check input/output edges are incident to the subgraph
        for q, e in self.input_edges.items():
            if e not in valid_io:
                errors.append(f"Input edge {e} (qubit {q}) not incident to subgraph")
        for q, e in self.output_edges.items():
            if e not in valid_io:
                errors.append(f"Output edge {e} (qubit {q}) not incident to subgraph")

        # Non-input/output boundary edges must not have qubit labels
        io_edges = set(self.input_edges.values()) | set(self.output_edges.values())
        for e in boundary - io_edges:
            if e in self.edge_labels:
                errors.append(
                    f"Boundary edge {e} is neither input nor output "
                    f"but has qubit label {self.edge_labels[e]}")

        # At most 2 incident edges per qubit per vertex
        edge_qubits: Dict[ET, list[int]] = {}
        for q, e in self.input_edges.items():
            edge_qubits.setdefault(e, []).append(q)
        for q, e in self.output_edges.items():
            edge_qubits.setdefault(e, []).append(q)
        for e, q in self.edge_labels.items():
            edge_qubits.setdefault(e, []).append(q)

        for v in self.vertices:
            label_counts: Dict[int, int] = {}
            for e in self.g.incident_edges(v):
                for q in edge_qubits.get(e, []):
                    label_counts[q] = label_counts.get(q, 0) + 1
            for q, count in label_counts.items():
                if count > 2:
                    errors.append(
                        f"Vertex {v} has {count} incident edges labeled "
                        f"qubit {q} (max 2)")

        return errors


@dataclass
class CliffordUnitary(CircuitLike):
    """A CircuitLike known to be a Clifford gate (H, S, CNOT, etc.)."""
    pass


@dataclass
class PauliBox(CircuitLike):
    """A CircuitLike known to be a Pauli box."""
    pauli_string: str = ""


# ---------------------------------------------------------------------------
# Adjacency and composition
# ---------------------------------------------------------------------------

def check_same_support_adjacent(
    a: CircuitLike,
    b: CircuitLike,
) -> Optional[Dict[int, int]]:
    """Check if a's labeled outputs are exactly b's labeled inputs.

    Returns a qubit mapping {a_qubit: b_qubit} if adjacent, else None.
    """
    a_out = a.output_qubit_map()
    b_in = b.input_qubit_map()

    a_out_edges = set(a_out.values())
    b_in_edges = set(b_in.values())
    if a_out_edges != b_in_edges:
        return None

    b_in_by_edge = {e: q for q, e in b_in.items()}
    mapping: Dict[int, int] = {}
    for a_qubit, edge in a_out.items():
        b_qubit = b_in_by_edge[edge]
        mapping[a_qubit] = b_qubit

    return mapping


def vertical_compose(a: CircuitLike, b: CircuitLike) -> CircuitLike:
    """Vertical composition (tensor product) of two CircuitLikes.

    B's qubit labels are shifted to avoid collision with A's when needed.
    Both must reference the same parent graph.
    Raises ValueError if vertex sets overlap.
    """
    overlap = a.vertices & b.vertices
    if overlap:
        raise ValueError(f"Vertex sets overlap: {overlap}")

    a_qubits = (set(a.input_edges.keys()) | set(a.output_edges.keys())
                | set(a.edge_labels.values()))
    b_qubits = (set(b.input_edges.keys()) | set(b.output_edges.keys())
                | set(b.edge_labels.values()))

    shift = max(a_qubits) + 1 if a_qubits & b_qubits else 0

    input_edges = dict(a.input_edges)
    for q, e in b.input_edges.items():
        input_edges[q + shift] = e

    output_edges = dict(a.output_edges)
    for q, e in b.output_edges.items():
        output_edges[q + shift] = e

    edge_labels = dict(a.edge_labels)
    for e, q in b.edge_labels.items():
        edge_labels[e] = q + shift

    return CircuitLike(
        vertices=a.vertices | b.vertices,
        g=a.g,
        input_edges=input_edges,
        output_edges=output_edges,
        edge_labels=edge_labels,
    )


def horizontal_compose(
    a: CircuitLike,
    b: CircuitLike,
) -> Optional[CircuitLike]:
    """Horizontal composition (sequential / function composition).

    A's outputs must equal B's inputs (as edge sets).  Shared edges
    become internal.  The result uses A's qubit labeling.

    Returns None if the two are not same-support adjacent.
    """
    qubit_map = check_same_support_adjacent(a, b)
    if qubit_map is None:
        return None

    inv_map = {v: k for k, v in qubit_map.items()}

    input_edges = dict(a.input_edges)

    output_edges: Dict[int, ET] = {}
    for b_q, e in b.output_edges.items():
        a_q = inv_map.get(b_q, b_q)
        output_edges[a_q] = e

    edge_labels = dict(a.edge_labels)
    for e, b_q in b.edge_labels.items():
        a_q = inv_map.get(b_q, b_q)
        edge_labels[e] = a_q

    # Shared edges (former A outputs / B inputs) become internal
    for a_q, shared_e in a.output_edges.items():
        edge_labels[shared_e] = a_q

    return CircuitLike(
        vertices=a.vertices | b.vertices,
        g=a.g,
        input_edges=input_edges,
        output_edges=output_edges,
        edge_labels=edge_labels,
    )
