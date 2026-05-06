"""CircuitLike: a typed subgraph with labeled input/output edges.

A CircuitLike represents a region of a ZX graph that behaves like a
circuit — it has input and output wires, each labeled with a qubit
integer.  Edges can also carry qubit labels internally (along a wire),
while cross-qubit interaction edges and special boundary edges (like a
Pauli box's extra leg) remain unlabeled.

Constraint: each vertex has at most 2 incident edges with the same
qubit label (one "in", one "out" along the wire).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set, Dict, Iterable

from pyzx.graph.base import BaseGraph, VT, ET


@dataclass
class CircuitLike:
    """A subgraph with classified and labeled boundary edges.

    Attributes:
        vertices: The vertex IDs belonging to this subgraph.
        input_edges: Boundary edges classified as inputs.
        output_edges: Boundary edges classified as outputs.
        edge_labels: Mapping from edge to qubit label.  Applies to any
            edge (input, output, or internal).  Edges absent from this
            dict are unlabeled (cross-qubit interactions, extra legs, etc.).
    """
    vertices: Set[VT]
    input_edges: Set[ET] = field(default_factory=set)
    output_edges: Set[ET] = field(default_factory=set)
    edge_labels: Dict[ET, int] = field(default_factory=dict)

    # --- derived helpers ---------------------------------------------------

    def boundary_edges(self, g: BaseGraph) -> Set[ET]:
        """All edges crossing the subgraph boundary."""
        vset = self.vertices
        result: Set[ET] = set()
        for v in vset:
            for e in g.incident_edges(v):
                s, t = g.edge_st(e)
                other = t if s == v else s
                if other not in vset:
                    result.add(e)
        return result

    def neutral_edges(self, g: BaseGraph) -> Set[ET]:
        """Boundary edges that are neither input nor output."""
        return self.boundary_edges(g) - self.input_edges - self.output_edges

    def internal_edges(self, g: BaseGraph) -> Set[ET]:
        """Edges with both endpoints inside the subgraph."""
        vset = self.vertices
        result: Set[ET] = set()
        for v in vset:
            for e in g.incident_edges(v):
                s, t = g.edge_st(e)
                if s in vset and t in vset:
                    result.add(e)
        return result

    def input_qubit_map(self) -> Dict[int, ET]:
        """Map qubit label → input edge, for all labeled input edges."""
        return {self.edge_labels[e]: e
                for e in self.input_edges
                if e in self.edge_labels}

    def output_qubit_map(self) -> Dict[int, ET]:
        """Map qubit label → output edge, for all labeled output edges."""
        return {self.edge_labels[e]: e
                for e in self.output_edges
                if e in self.edge_labels}

    # --- validation --------------------------------------------------------

    def validate(self, g: BaseGraph) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: list[str] = []

        # Check input/output edges are actually boundary edges
        boundary = self.boundary_edges(g)
        for e in self.input_edges:
            if e not in boundary:
                errors.append(f"Input edge {e} is not a boundary edge")
        for e in self.output_edges:
            if e not in boundary:
                errors.append(f"Output edge {e} is not a boundary edge")

        # Input and output sets must be disjoint
        overlap = self.input_edges & self.output_edges
        if overlap:
            errors.append(f"Edges classified as both input and output: {overlap}")

        # At most 2 incident edges per qubit per vertex
        for v in self.vertices:
            label_counts: Dict[int, int] = {}
            for e in g.incident_edges(v):
                if e in self.edge_labels:
                    q = self.edge_labels[e]
                    label_counts[q] = label_counts.get(q, 0) + 1
            for q, count in label_counts.items():
                if count > 2:
                    errors.append(
                        f"Vertex {v} has {count} incident edges labeled qubit {q} (max 2)")

        return errors


def check_same_support_adjacent(
    a: CircuitLike,
    b: CircuitLike,
) -> Optional[Dict[int, int]]:
    """Check if a's labeled outputs are exactly b's labeled inputs.

    Returns a qubit mapping {a_qubit: b_qubit} if adjacent, else None.
    The mapping pairs a's output qubit labels with b's input qubit labels
    based on shared edges.
    """
    a_out = a.output_qubit_map()
    b_in = b.input_qubit_map()

    # The edge sets must match
    a_out_edges = set(a_out.values())
    b_in_edges = set(b_in.values())
    if a_out_edges != b_in_edges:
        return None

    # Build qubit mapping via shared edges
    b_in_by_edge = {e: q for q, e in b_in.items()}
    mapping: Dict[int, int] = {}
    for a_qubit, edge in a_out.items():
        b_qubit = b_in_by_edge[edge]
        mapping[a_qubit] = b_qubit

    return mapping
