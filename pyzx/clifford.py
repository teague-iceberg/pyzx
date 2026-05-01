"""Predicates for Clifford-fragment diagrams."""

from __future__ import annotations

import cmath
from math import sqrt
from typing import TYPE_CHECKING, Iterable

from .symbolic import Poly
from .tensor import is_unitary
from .utils import (
    EdgeType,
    VertexType,
    get_h_box_label,
    get_w_io,
    get_z_box_label,
    phase_is_clifford,
    vertex_is_w,
)

if TYPE_CHECKING:
    from .graph.base import BaseGraph, VT


DEFAULT_TOLERANCE = 1e-9
SQRT_HALF = 1 / sqrt(2)
EIGHTH_ROOTS = (
    1,
    SQRT_HALF + SQRT_HALF * 1j,
    1j,
    -SQRT_HALF + SQRT_HALF * 1j,
    -1,
    -SQRT_HALF - SQRT_HALF * 1j,
    -1j,
    SQRT_HALF - SQRT_HALF * 1j,
)


def graph_is_in_clifford_fragment(g: BaseGraph, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """Return whether all vertices of ``g`` are in the ZXHW Clifford fragment.

    This is a syntactic/local predicate for the generator set used by PyZX:

    * Z/X spiders must have phase an integer multiple of pi/2.
    * Z-box labels must be 0 or one of 1, i, -1, -i.
    * W nodes must have at most two external legs.
    * H-box labels must satisfy the arity-dependent Clifford cases:
      zero legs allow 0 and eighth roots of unity; one leg allows
      0, 1, i, -1, -i; two legs allow 1, -1; higher arities allow only 1.

    It is not a complete semantic test for whether an arbitrary diagram
    denotes a Clifford unitary after cancellations. For example, a graph
    containing locally non-Clifford generators can still simplify to a
    Clifford map, but this predicate will reject it.
    """
    seen_w: set[VT] = set()
    for v in g.vertices():
        ty = g.type(v)
        if ty == VertexType.BOUNDARY:
            continue
        if ty in (VertexType.Z, VertexType.X):
            if not phase_is_clifford(g.phase(v)):
                return False
        elif ty == VertexType.Z_BOX:
            if not _complex_is_close_to_any(
                get_z_box_label(g, v),
                (0, 1, 1j, -1, -1j),
                tolerance,
            ):
                return False
        elif ty == VertexType.H_BOX:
            if not _h_box_is_in_clifford_fragment(g, v, tolerance):
                return False
        elif vertex_is_w(ty):
            if v in seen_w:
                continue
            try:
                w_in, w_out = get_w_io(g, v)
            except AssertionError:
                return False
            seen_w.add(w_in)
            seen_w.add(w_out)
            if _w_node_external_legs(g, w_in, w_out) > 2:
                return False
        else:
            return False
    return True


def graph_is_unitary_in_clifford_fragment(g: BaseGraph, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """Return whether ``g`` is unitary and uses only Clifford-fragment generators.

    This combines PyZX's existing unitary check, which works up to nonzero
    scalar, with :func:`graph_is_in_clifford_fragment`. It deliberately does
    not try to prove that a non-Clifford-looking graph semantically simplifies
    to a Clifford unitary.
    """
    if g.num_inputs() != g.num_outputs():
        return False
    if not graph_is_in_clifford_fragment(g, tolerance):
        return False
    return is_unitary(g)


def _h_box_is_in_clifford_fragment(g: BaseGraph, v: VT, tolerance: float) -> bool:
    legs = _external_legs(g, v)

    label = g.vdata(v, "label", None)
    if label is not None:
        return _h_box_label_is_in_clifford_fragment(label, legs, tolerance)

    phase = g.phase(v)
    if isinstance(phase, Poly):
        return _h_box_legacy_phase_is_in_clifford_fragment(phase, legs)

    try:
        label = get_h_box_label(g, v)
    except ValueError:
        return False
    return _h_box_label_is_in_clifford_fragment(label, legs, tolerance)


def _h_box_label_is_in_clifford_fragment(label: object, legs: int, tolerance: float) -> bool:
    if legs == 0:
        # A 0-legged H-box denotes the scalar given by its label. The full
        # Clifford scalar ring is larger than this finite set, but recognizing
        # it from arbitrary complex floats would be an approximate semantic
        # scalar test. For this local fragment predicate, use the arity pattern:
        # 0 legs allow quarter-turn phases, then 1 leg allows half-turn phases,
        # 2 legs allow integer phases, and 3+ legs allow even-integer phases.
        return _complex_is_close_to_any(label, (0, *EIGHTH_ROOTS), tolerance)
    if legs == 1:
        return _complex_is_close_to_any(label, (0, 1, 1j, -1, -1j), tolerance)
    if legs == 2:
        return _complex_is_close_to_any(label, (1, -1), tolerance)
    return _complex_is_close_to_any(label, (1,), tolerance)


def _h_box_legacy_phase_is_in_clifford_fragment(phase: Poly, legs: int) -> bool:
    # Legacy H-box phases are multiples of pi and denote the complex label
    # exp(i*pi*phase). Symbolic phase polynomials can be accepted when they
    # always evaluate to an arity-dependent allowed phase:
    #   0 legs: multiples of 1/4
    #   1 leg: multiples of 1/2
    #   2 legs: integers
    #   3+ legs: even integers
    if legs == 0:
        return _poly_has_bool_vars_and_denominator(phase, 4)
    if legs == 1:
        return phase.is_clifford
    if legs == 2:
        return phase.is_pauli
    return phase.is_pauli and (phase % 2) == 0


def _poly_has_bool_vars_and_denominator(poly: Poly, denominator: int) -> bool:
    for coefficient, term in poly.terms:
        if isinstance(coefficient, complex):
            return False
        if not all(var.is_bool for var, _ in term.vars):
            return False
        if coefficient * denominator % 1 != 0:
            return False
    return True


def _complex_is_close_to_any(value: object, allowed: Iterable[complex], tolerance: float) -> bool:
    # Z-box labels may be symbolic Poly objects. For those, use exact
    # symbolic equality against the small list of allowed constants instead
    # of trying to coerce to complex.
    if isinstance(value, Poly):
        return any(_poly_equals_constant(value, candidate) for candidate in allowed)
    try:
        z = complex(value)
    except (TypeError, ValueError):
        return False
    return any(cmath.isclose(z, complex(candidate), abs_tol=tolerance) for candidate in allowed)


def _poly_equals_constant(poly: Poly, value: complex) -> bool:
    simplified = poly + 0
    if not simplified.terms:
        return value == 0
    if len(simplified.terms) != 1:
        return False
    coefficient, term = simplified.terms[0]
    return not term.vars and coefficient == value


def _external_legs(g: BaseGraph, v: VT) -> int:
    """Count tensor legs incident to a single graph vertex.

    This is not always the same as the number of neighbours: a self-loop is
    two tensor indices on the same generator, so it contributes two legs.
    """
    legs = 0
    for e in g.incident_edges(v):
        legs += 2 if g.edge_s(e) == g.edge_t(e) else 1
    return legs


def _w_node_external_legs(g: BaseGraph, w_in: VT, w_out: VT) -> int:
    """Count external tensor legs of a conceptual W node.

    PyZX represents one W node as a W_INPUT/W_OUTPUT pair joined by an
    internal W_IO edge. That internal edge is part of the representation, not
    an external leg of the W tensor, so it is skipped here. Normal PyZX graph
    construction rejects W self-loops, but counting them as two legs keeps this
    helper consistent with the tensor-leg convention used for other vertices.
    """
    legs = 0
    for v in (w_in, w_out):
        for e in g.incident_edges(v):
            if g.edge_type(e) == EdgeType.W_IO:
                continue
            legs += 2 if g.edge_s(e) == g.edge_t(e) else 1
    return legs
