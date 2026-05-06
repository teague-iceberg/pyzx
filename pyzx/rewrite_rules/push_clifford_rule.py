"""Rewrite rule: push a Clifford gate through a Pauli box.

Given a Clifford unitary C adjacent to a Pauli box P (same support),
this rule conjugates P by C (modifying conjugation gates in place)
and moves C to the other side of P.

For V1, the Clifford must be a single generator gate (H, S, or CNOT).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.utils import EdgeType, VertexType, set_h_box_label
from pyzx.circuitlike import CircuitLike, check_same_support_adjacent
from pyzx.paulibox import (
    generate_pauli_box,
    conjugate_pauli_string,
    identify_clifford_gate,
    identify_pauli_box,
    _trace_wire,
)


@dataclass
class PushCliffordMatch:
    """Information about a valid Clifford-through-Pauli-box push."""
    clifford: CircuitLike
    pauli: CircuitLike
    qubit_map: dict[int, int]    # clifford qubit → pauli qubit
    clifford_name: str           # "H", "S", or "CNOT"
    clifford_wire: int           # target wire (in pauli qubit labels)
    clifford_control: Optional[int]  # control wire for CNOT
    pauli_string: str


def check_push_clifford(
    g: BaseGraph,
    clifford: CircuitLike,
    pauli: CircuitLike,
) -> Optional[PushCliffordMatch]:
    """Check if a Clifford gate can be pushed through a Pauli box.

    Returns a PushCliffordMatch if the push is valid, else None.
    """
    qubit_map = check_same_support_adjacent(clifford, pauli)
    if qubit_map is None:
        return None

    cliff_info = identify_clifford_gate(g, clifford)
    if cliff_info is None:
        return None
    cliff_name, cliff_wire, cliff_control = cliff_info

    pauli_string = identify_pauli_box(g, pauli)
    if pauli_string is None:
        return None

    pauli_wire = qubit_map[cliff_wire]
    pauli_control = qubit_map[cliff_control] if cliff_control is not None else None

    return PushCliffordMatch(
        clifford=clifford,
        pauli=pauli,
        qubit_map=qubit_map,
        clifford_name=cliff_name,
        clifford_wire=pauli_wire,
        clifford_control=pauli_control,
        pauli_string=pauli_string,
    )


@dataclass
class PushCliffordResult:
    """Result of applying the push-Clifford rule."""
    new_pauli_vertices: set[VT]
    new_clifford_vertices: set[VT]
    new_pauli_string: str


def apply_push_clifford(
    g: BaseGraph,
    match: PushCliffordMatch,
) -> PushCliffordResult:
    """Apply the push: modify the Pauli box's conjugation gates in place
    and move the Clifford to the other side.

    The Pauli box stays in its current position. Only the conjugation
    gates on the affected wire(s) are changed to reflect the new Pauli
    letter.  The Clifford gate is removed from the input side and
    re-inserted on the output side.

    Mutates g in place. Returns info about the new vertices.
    """
    clifford = match.clifford
    pauli = match.pauli
    old_ps = match.pauli_string
    new_ps = conjugate_pauli_string(
        old_ps,
        match.clifford_name,
        match.clifford_wire,
        match.clifford_control,
    )

    # --- Step 1: Snapshot Clifford's upstream connection ---
    cliff_upstream: dict[int, tuple[VT, EdgeType]] = {}
    for qubit, edge in clifford.input_qubit_map().items():
        s, t = g.edge_st(edge)
        outside = t if s in clifford.vertices else s
        cliff_upstream[qubit] = (outside, g.edge_type(edge))

    # --- Step 2: Snapshot Pauli box's downstream connections ---
    pauli_downstream: dict[int, tuple[VT, EdgeType]] = {}
    for qubit, edge in pauli.output_qubit_map().items():
        s, t = g.edge_st(edge)
        outside = t if s in pauli.vertices else s
        pauli_downstream[qubit] = (outside, g.edge_type(edge))

    # --- Step 3: For each affected wire, find the conjugation vertices ---
    pauli_in_map = pauli.input_qubit_map()
    pauli_out_map = pauli.output_qubit_map()

    # Find the X spider (central hub)
    x_spider = None
    for v in pauli.vertices:
        if g.type(v) == VertexType.X and g.phase(v) == 0:
            # Check it's connected to Z spiders (not a conjugation gate)
            z_neighbors = [w for w in g.neighbors(v)
                          if w in pauli.vertices and g.type(w) == VertexType.Z]
            if z_neighbors:
                x_spider = v
                break

    # For each qubit wire in the Pauli box, trace the wire to find vertices
    wire_info: dict[int, list[VT]] = {}
    for qubit in sorted(pauli_in_map.keys()):
        in_edge = pauli_in_map[qubit]
        out_edge = pauli_out_map[qubit]
        wire_verts = _trace_wire(g, pauli, in_edge, out_edge)
        if wire_verts is not None:
            wire_info[qubit] = wire_verts

    # --- Step 4: Remove the Clifford gate vertices ---
    # First disconnect from Pauli box input
    shared_edge = None
    for qubit, edge in clifford.output_qubit_map().items():
        shared_edge = edge

    # Remove Clifford vertices
    for v in list(clifford.vertices):
        g.remove_vertex(v)

    # Reconnect: Clifford's upstream → Pauli box's first vertex on that wire
    for cliff_qubit, (up_v, up_ety) in cliff_upstream.items():
        pauli_qubit = match.qubit_map[cliff_qubit]
        if pauli_qubit in wire_info and wire_info[pauli_qubit]:
            first_on_wire = wire_info[pauli_qubit][0]
            g.add_edge((up_v, first_on_wire), up_ety)

    # --- Step 5: Modify conjugation gates in place ---
    for qubit in sorted(wire_info.keys()):
        if qubit >= len(old_ps) or qubit >= len(new_ps):
            continue
        old_letter = old_ps[qubit]
        new_letter = new_ps[qubit]
        if old_letter == new_letter:
            continue

        wire_verts = wire_info[qubit]
        # Find the Z spider on this wire (connected to X spider)
        z_spider = None
        for v in wire_verts:
            if g.type(v) == VertexType.Z and g.phase(v) == 0:
                if x_spider is not None and g.connected(v, x_spider):
                    z_spider = v
                    break
        if z_spider is None:
            continue

        z_idx = wire_verts.index(z_spider)
        pre_verts = wire_verts[:z_idx]
        post_verts = wire_verts[z_idx + 1:]

        _update_conjugation(g, pre_verts, post_verts, z_spider,
                           old_letter, new_letter, qubit, pauli)

    # --- Step 6: Insert Clifford on the output side ---
    new_cliff_verts: set[VT] = set()
    for cliff_qubit, (up_v, up_ety) in cliff_upstream.items():
        pauli_qubit = match.qubit_map[cliff_qubit]
        if pauli_qubit not in pauli_downstream:
            continue

        down_v, down_ety = pauli_downstream[pauli_qubit]

        # Find the last Pauli box vertex on this wire
        if pauli_qubit in wire_info and wire_info[pauli_qubit]:
            last_on_wire = wire_info[pauli_qubit][-1]
        else:
            continue

        # Remove old output edge (last_on_wire → down_v)
        for e in list(g.edges(last_on_wire, down_v)):
            g.remove_edge(e)

        # Position: midpoint between last_on_wire and down_v
        cliff_row = (g.row(last_on_wire) + g.row(down_v)) / 2
        cliff_qubit_pos = g.qubit(last_on_wire)

        if match.clifford_name == "H":
            h = g.add_vertex(VertexType.H_BOX,
                             qubit=cliff_qubit_pos, row=cliff_row)
            set_h_box_label(g, h, -1)
            g.add_edge((last_on_wire, h), EdgeType.SIMPLE)
            g.add_edge((h, down_v), down_ety)
            new_cliff_verts.add(h)
        elif match.clifford_name == "S":
            s_v = g.add_vertex(VertexType.Z, phase=Fraction(1, 2),
                               qubit=cliff_qubit_pos, row=cliff_row)
            g.add_edge((last_on_wire, s_v), EdgeType.SIMPLE)
            g.add_edge((s_v, down_v), down_ety)
            new_cliff_verts.add(s_v)

    # Collect the updated Pauli box vertices (original + any inserted conjugation gates)
    new_pauli_verts = set(pauli.vertices)
    # Add any new vertices created by _insert_conjugation_around
    # (they won't be in pauli.vertices yet)
    for qubit, verts in wire_info.items():
        for v in verts:
            if v in g.vertices():
                new_pauli_verts.add(v)
    if x_spider is not None and x_spider in g.vertices():
        new_pauli_verts.add(x_spider)
    # Remove any vertices that were deleted during conjugation update
    new_pauli_verts = {v for v in new_pauli_verts if v in g.vertices()}

    return PushCliffordResult(
        new_pauli_vertices=new_pauli_verts,
        new_clifford_vertices=new_cliff_verts,
        new_pauli_string=new_ps,
    )


def _update_conjugation(
    g: BaseGraph,
    pre_verts: list[VT],
    post_verts: list[VT],
    z_spider: VT,
    old_letter: str,
    new_letter: str,
    qubit: int,
    pauli: CircuitLike,
) -> None:
    """Update conjugation gates on a wire when the Pauli letter changes.

    Modifies vertices in place where possible, inserts/removes as needed.
    """
    # Determine what conjugation each letter needs:
    # Z: none
    # X: H_BOX before and after
    # Y: X(π/2) before and X(3π/2) after
    # I: none (but wire shouldn't be connected to X spider — not handled here)

    def _set_as_h_box(v: VT) -> None:
        g.set_type(v, VertexType.H_BOX)
        g.set_phase(v, 0)
        set_h_box_label(g, v, -1)

    def _set_as_sqrt_x(v: VT) -> None:
        g.set_type(v, VertexType.X)
        g.set_phase(v, Fraction(1, 2))

    def _set_as_sqrt_x_dag(v: VT) -> None:
        g.set_type(v, VertexType.X)
        g.set_phase(v, Fraction(3, 2))

    # Cases where we can modify in place (same number of conjugation vertices)
    if len(pre_verts) == 1 and len(post_verts) == 1:
        # Both old and new have 1 conjugation gate each side
        if new_letter == 'X':
            _set_as_h_box(pre_verts[0])
            _set_as_h_box(post_verts[0])
        elif new_letter == 'Y':
            _set_as_sqrt_x(pre_verts[0])
            _set_as_sqrt_x_dag(post_verts[0])
        elif new_letter == 'Z':
            # Remove conjugation gates: wire Z spider directly
            _remove_and_bypass(g, pre_verts[0], pauli)
            _remove_and_bypass(g, post_verts[0], pauli)

    elif len(pre_verts) == 0 and len(post_verts) == 0:
        # Old was Z (no conjugation), new needs conjugation
        if new_letter in ('X', 'Y'):
            _insert_conjugation_around(g, z_spider, new_letter)


def _remove_and_bypass(g: BaseGraph, v: VT, pauli: CircuitLike) -> None:
    """Remove vertex v and connect its two neighbors directly."""
    neighbors = list(g.neighbors(v))
    if len(neighbors) != 2:
        return
    n0, n1 = neighbors
    # Determine edge types
    ety0 = g.edge_type(next(g.edges(v, n0)))
    ety1 = g.edge_type(next(g.edges(v, n1)))
    # Combined edge type
    if ety0 == EdgeType.HADAMARD and ety1 == EdgeType.HADAMARD:
        combined = EdgeType.SIMPLE
    elif ety0 == EdgeType.HADAMARD or ety1 == EdgeType.HADAMARD:
        combined = EdgeType.HADAMARD
    else:
        combined = EdgeType.SIMPLE
    g.remove_vertex(v)
    g.add_edge((n0, n1), combined)
    pauli.vertices.discard(v)


def _insert_conjugation_around(
    g: BaseGraph,
    z_spider: VT,
    new_letter: str,
) -> None:
    """Insert conjugation gates before and after a Z spider on its wire."""
    qubit = g.qubit(z_spider)
    row = g.row(z_spider)

    # Find the wire neighbors of the Z spider (not the X spider hub)
    wire_neighbors = []
    hub_neighbors = []
    for w in g.neighbors(z_spider):
        if g.type(w) == VertexType.X and g.phase(w) == 0:
            hub_neighbors.append(w)
        else:
            wire_neighbors.append(w)

    if len(wire_neighbors) != 2:
        return

    # Sort by row to determine input/output direction
    wire_neighbors.sort(key=lambda w: g.row(w))
    pre_neighbor, post_neighbor = wire_neighbors

    # Remove edges to wire neighbors
    for e in list(g.edges(z_spider, pre_neighbor)):
        g.remove_edge(e)
    for e in list(g.edges(z_spider, post_neighbor)):
        g.remove_edge(e)

    if new_letter == 'X':
        pre = g.add_vertex(VertexType.H_BOX, qubit=qubit, row=row - 0.5)
        set_h_box_label(g, pre, -1)
        post = g.add_vertex(VertexType.H_BOX, qubit=qubit, row=row + 0.5)
        set_h_box_label(g, post, -1)
    elif new_letter == 'Y':
        pre = g.add_vertex(VertexType.X, qubit=qubit, row=row - 0.5,
                          phase=Fraction(1, 2))
        post = g.add_vertex(VertexType.X, qubit=qubit, row=row + 0.5,
                           phase=Fraction(3, 2))
    else:
        return

    g.add_edge((pre_neighbor, pre), EdgeType.SIMPLE)
    g.add_edge((pre, z_spider), EdgeType.SIMPLE)
    g.add_edge((z_spider, post), EdgeType.SIMPLE)
    g.add_edge((post, post_neighbor), EdgeType.SIMPLE)
