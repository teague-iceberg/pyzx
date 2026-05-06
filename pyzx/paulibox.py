"""Pauli box generation, identification, and conjugation.

A Pauli box for an n-qubit Pauli string P = P_0 P_1 ... P_{n-1} is a
ZX subdiagram with:
  - n qubit wires, each containing a phaseless Z spider
  - Conjugation gates on the wire before/after the Z spider to
    implement non-Z Paulis (H for X, HSdg for Y)
  - All Z spiders connected to a single phaseless X spider
  - The X spider has one extra output edge (the "phase leg")
  - 2n labeled boundary edges (n inputs + n outputs) plus
    1 unlabeled boundary edge (the phase leg)
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.utils import EdgeType, VertexType, set_h_box_label
from pyzx.circuitlike import CircuitLike


def generate_pauli_box(
    pauli_string: str,
    g: Optional[BaseGraph] = None,
    start_row: float = 0,
    start_qubit: float = 0,
) -> tuple[BaseGraph, CircuitLike]:
    """Generate a Pauli box for the given Pauli string.

    Args:
        pauli_string: e.g. "XZIY" — one character per qubit.
            I = identity (wire passes through Z spider, which acts as identity
                when connected to the central X spider — included for
                uniform structure)
            X = Hadamard-conjugated Z
            Y = HSdg-conjugated Z
            Z = bare Z
        g: Graph to add vertices to.  If None, creates a new Multigraph.
        start_row: Row offset for positioning.
        start_qubit: Qubit offset for positioning.

    Returns:
        (graph, circuit_like) — the graph and its CircuitLike annotation.
    """
    if g is None:
        from pyzx.graph.multigraph import Multigraph
        g = Multigraph()
        g.set_auto_simplify(False)

    n = len(pauli_string)
    input_edges: set[ET] = set()
    output_edges: set[ET] = set()
    edge_labels: dict[ET, int] = {}
    vertices: set[VT] = set()

    # Strict column layout:
    #   col 0: input boundaries
    #   col 1: pre-conjugation gates
    #   col 2: Z phaseless spiders
    #   col 3: post-conjugation gates (and X spider + phase boundary above)
    #   col 4: output boundaries
    col_in = start_row
    col_pre = start_row + 1
    col_z = start_row + 2
    col_post = start_row + 3
    col_out = start_row + 4

    z_spiders: list[VT] = []

    for i, pauli in enumerate(pauli_string.upper()):
        q = start_qubit + i

        # Input boundary (col 0)
        b_in = g.add_vertex(VertexType.BOUNDARY, qubit=q, row=col_in)

        if pauli == 'I':
            # Identity wire: straight through, no Z spider, no X connection
            b_out = g.add_vertex(VertexType.BOUNDARY, qubit=q, row=col_out)
            _add_simple_edge(g, b_in, b_out)
            # No internal vertices on this wire, so no input/output edges
            # from the subgraph's perspective
            continue

        # Pre-conjugation (col 1)
        prev = b_in
        if pauli == 'X':
            h = g.add_vertex(VertexType.H_BOX, qubit=q, row=col_pre)
            set_h_box_label(g, h, -1)
            vertices.add(h)
            e = _add_simple_edge(g, prev, h)
            edge_labels[e] = i
            prev = h
        elif pauli == 'Y':
            sx = g.add_vertex(VertexType.X, qubit=q, row=col_pre, phase=Fraction(1, 2))
            vertices.add(sx)
            e = _add_simple_edge(g, prev, sx)
            edge_labels[e] = i
            prev = sx
        # Z: no pre-conjugation, wire goes straight to Z spider

        # Z spider (col 2)
        z = g.add_vertex(VertexType.Z, qubit=q, row=col_z, phase=0)
        vertices.add(z)
        e_to_z = _add_simple_edge(g, prev, z)
        edge_labels[e_to_z] = i
        z_spiders.append(z)
        prev = z

        # Post-conjugation (col 3)
        if pauli == 'X':
            h2 = g.add_vertex(VertexType.H_BOX, qubit=q, row=col_post)
            set_h_box_label(g, h2, -1)
            vertices.add(h2)
            e = _add_simple_edge(g, prev, h2)
            edge_labels[e] = i
            prev = h2
        elif pauli == 'Y':
            sxdg = g.add_vertex(VertexType.X, qubit=q, row=col_post, phase=Fraction(3, 2))
            vertices.add(sxdg)
            e = _add_simple_edge(g, prev, sxdg)
            edge_labels[e] = i
            prev = sxdg

        # Output boundary (col 4)
        b_out = g.add_vertex(VertexType.BOUNDARY, qubit=q, row=col_out)

        # Input/output edges
        e_in = list(g.edges(b_in, next(_first_neighbor(g, b_in))))[0]
        e_out = _add_simple_edge(g, prev, b_out)
        input_edges.add(e_in)
        output_edges.add(e_out)
        edge_labels[e_in] = i
        edge_labels[e_out] = i

    # Central X spider at col_post, above the top wire
    x_spider = g.add_vertex(VertexType.X, qubit=start_qubit - 1,
                            row=col_post, phase=0)
    vertices.add(x_spider)

    # Connect all Z spiders to the central X spider
    for z in z_spiders:
        g.add_edge((z, x_spider), EdgeType.SIMPLE)

    # Phase leg boundary above the X spider
    phase_boundary = g.add_vertex(VertexType.BOUNDARY,
                                  qubit=start_qubit - 2,
                                  row=col_post)
    g.add_edge((x_spider, phase_boundary), EdgeType.SIMPLE)

    cl = CircuitLike(
        vertices=vertices,
        input_edges=input_edges,
        output_edges=output_edges,
        edge_labels=edge_labels,
    )

    return g, cl


def identify_clifford_gate(
    g: BaseGraph,
    cl: CircuitLike,
) -> Optional[tuple[str, int, Optional[int]]]:
    """Identify a CircuitLike as a single Clifford generator gate.

    Returns (gate_name, target_wire, control_wire) or None.
    gate_name is "H", "S", or "CNOT".
    Wire indices are in the CircuitLike's own qubit labels.
    """
    verts = cl.vertices
    labeled_in = cl.input_qubit_map()
    labeled_out = cl.output_qubit_map()

    # Single-qubit gates: exactly 1 internal vertex, 1 qubit
    if len(labeled_in) == 1 and len(labeled_out) == 1:
        qubit = next(iter(labeled_in.keys()))
        internal = [v for v in verts if g.type(v) != VertexType.BOUNDARY]
        if len(internal) != 1:
            return None
        v = internal[0]

        # Hadamard: H-box vertex, or phaseless Z with Hadamard edge
        if g.type(v) == VertexType.H_BOX:
            return ("H", qubit, None)
        if g.type(v) == VertexType.Z:
            in_edge = labeled_in[qubit]
            out_edge = labeled_out[qubit]
            phase = g.phase(v)
            if phase == 0:
                had_count = sum(1 for e in [in_edge, out_edge]
                               if g.edge_type(e) == EdgeType.HADAMARD)
                if had_count >= 1:
                    return ("H", qubit, None)
            # S gate: Z spider with phase pi/2
            if phase == Fraction(1, 2):
                return ("S", qubit, None)

        return None

    # Two-qubit CNOT: Z spider connected to X spider
    if len(labeled_in) == 2 and len(labeled_out) == 2:
        internal = [v for v in verts if g.type(v) != VertexType.BOUNDARY]
        if len(internal) != 2:
            return None

        z_verts = [v for v in internal if g.type(v) == VertexType.Z and g.phase(v) == 0]
        x_verts = [v for v in internal if g.type(v) == VertexType.X and g.phase(v) == 0]

        if len(z_verts) != 1 or len(x_verts) != 1:
            return None

        z, x = z_verts[0], x_verts[0]

        # They must be connected by a simple edge
        if not g.connected(z, x):
            return None
        zx_edges = list(g.edges(z, x))
        if len(zx_edges) != 1 or g.edge_type(zx_edges[0]) != EdgeType.SIMPLE:
            return None

        # Find which qubit each is on by checking which labeled edges
        # are incident to each vertex
        z_qubit = None
        x_qubit = None
        for qubit, edge in labeled_in.items():
            s, t = g.edge_st(edge)
            if s == z or t == z:
                z_qubit = qubit
            elif s == x or t == x:
                x_qubit = qubit
        for qubit, edge in labeled_out.items():
            s, t = g.edge_st(edge)
            if s == z or t == z:
                z_qubit = qubit
            elif s == x or t == x:
                x_qubit = qubit

        if z_qubit is None or x_qubit is None:
            return None

        # CNOT: Z spider is control, X spider is target
        return ("CNOT", x_qubit, z_qubit)

    return None


def identify_pauli_box(
    g: BaseGraph,
    cl: CircuitLike,
) -> Optional[str]:
    """Identify a CircuitLike as a Pauli box and extract its Pauli string.

    Returns the Pauli string (e.g. "XZI") or None if not a valid Pauli box.

    For V1, this checks the structural pattern:
    - One phaseless X spider (the central spider)
    - The X spider has exactly n+1 edges: n to Z-type spiders, 1 extra (phase leg)
    - Each qubit wire passes through the structure
    """
    verts = cl.vertices
    labeled_in = cl.input_qubit_map()
    labeled_out = cl.output_qubit_map()
    n = len(labeled_in)

    if n == 0 or n != len(labeled_out):
        return None

    # Find the central X spider
    x_spiders = [v for v in verts
                 if g.type(v) == VertexType.X and g.phase(v) == 0]
    if len(x_spiders) != 1:
        return None
    x_center = x_spiders[0]

    # The X spider should connect to n phaseless Z spiders inside the subgraph
    x_neighbors_inside = [w for w in g.neighbors(x_center) if w in verts]
    z_connected = [w for w in x_neighbors_inside
                   if g.type(w) == VertexType.Z and g.phase(w) == 0]

    if len(z_connected) != n:
        return None

    # For each qubit wire, trace from input to output and determine the Pauli
    pauli_chars: dict[int, str] = {}
    for qubit in sorted(labeled_in.keys()):
        in_edge = labeled_in[qubit]
        out_edge = labeled_out[qubit]

        # Walk the wire from input edge to output edge, collecting vertices
        wire_verts = _trace_wire(g, cl, in_edge, out_edge)
        if wire_verts is None:
            return None

        # The wire must contain exactly one of the Z spiders connected to X center
        z_on_wire = [v for v in wire_verts if v in z_connected]
        if len(z_on_wire) != 1:
            return None

        # Determine Pauli type from conjugation gates on the wire
        pauli = _identify_pauli_from_wire(g, wire_verts, z_on_wire[0])
        if pauli is None:
            return None
        pauli_chars[qubit] = pauli

    return ''.join(pauli_chars[q] for q in sorted(pauli_chars.keys()))


def _trace_wire(
    g: BaseGraph,
    cl: CircuitLike,
    in_edge: ET,
    out_edge: ET,
) -> Optional[list[VT]]:
    """Trace a qubit wire through the subgraph from input to output edge.

    Returns the ordered list of internal vertices along the wire, or None
    if the trace fails.
    """
    # Start from the internal endpoint of the input edge
    s, t = g.edge_st(in_edge)
    current = s if s in cl.vertices else t

    result = [current]
    visited = {current}
    prev_edge = in_edge

    while True:
        # Find the next edge on this wire (the other labeled edge at this vertex,
        # or the output edge)
        found_next = False
        for e in g.incident_edges(current):
            if e == prev_edge:
                continue
            if e == out_edge:
                return result  # Reached the output
            es, et = g.edge_st(e)
            other = et if es == current else es
            if other in cl.vertices and other not in visited:
                # Check this is a wire edge (labeled with same qubit) or
                # could be part of the wire path
                if e in cl.edge_labels:
                    current = other
                    result.append(current)
                    visited.add(current)
                    prev_edge = e
                    found_next = True
                    break
        if not found_next:
            return None  # Wire trace failed

    return None


def _identify_pauli_from_wire(
    g: BaseGraph,
    wire_verts: list[VT],
    z_spider: VT,
) -> Optional[str]:
    """Identify the Pauli type from the conjugation pattern on a wire.

    Patterns (using H-boxes and √X gates):
    - Z: just the Z spider (no conjugation)
    - X: H_BOX — Z — H_BOX
    - Y: X(π/2) — Z — X(-π/2)  (√X conjugation)
    - I: just the Z spider (treated as Z)
    """
    z_idx = wire_verts.index(z_spider)

    before = wire_verts[:z_idx]
    after = wire_verts[z_idx + 1:]

    if len(before) == 0 and len(after) == 0:
        return 'Z'

    if len(before) == 1 and len(after) == 1:
        b, a = before[0], after[0]
        # X Pauli: H-box conjugation
        if g.type(b) == VertexType.H_BOX and g.type(a) == VertexType.H_BOX:
            return 'X'
        # Also accept old-style: phaseless Z spiders (with Hadamard edges)
        if (g.type(b) == VertexType.Z and g.phase(b) == 0 and
                g.type(a) == VertexType.Z and g.phase(a) == 0):
            return 'X'
        # Y Pauli: √X conjugation
        if (g.type(b) == VertexType.X and g.phase(b) == Fraction(1, 2) and
                g.type(a) == VertexType.X and g.phase(a) == Fraction(-1, 2)):
            return 'Y'
        # Also accept 3/2 as -1/2
        if (g.type(b) == VertexType.X and g.phase(b) == Fraction(1, 2) and
                g.type(a) == VertexType.X and g.phase(a) == Fraction(3, 2)):
            return 'Y'

    return None


def conjugate_pauli_string(
    pauli_string: str,
    clifford_name: str,
    wire: int,
    control_wire: Optional[int] = None,
) -> str:
    """Conjugate a Pauli string by a single Clifford gate on the given wire(s).

    Args:
        pauli_string: e.g. "XZI"
        clifford_name: "H", "S", or "CNOT"
        wire: The qubit index the gate acts on (target for CNOT).
        control_wire: For CNOT, the control qubit index.

    Returns:
        The conjugated Pauli string.
    """
    paulis = list(pauli_string.upper())

    if clifford_name == "H":
        # HXH = Z, HZH = X, HYH = -Y (sign absorbed into phase leg)
        p = paulis[wire]
        if p == 'X':
            paulis[wire] = 'Z'
        elif p == 'Z':
            paulis[wire] = 'X'
        # Y and I unchanged (up to sign)

    elif clifford_name == "S":
        # SXSdg = Y, SYSdg = -X, SZSdg = Z
        p = paulis[wire]
        if p == 'X':
            paulis[wire] = 'Y'
        elif p == 'Y':
            paulis[wire] = 'X'
        # Z and I unchanged (up to sign)

    elif clifford_name == "CNOT":
        assert control_wire is not None
        # CNOT conjugation rules:
        # X_c -> X_c X_t    (X on control spreads to target)
        # Z_t -> Z_c Z_t    (Z on target spreads to control)
        # X_t -> X_t         (X on target unchanged)
        # Z_c -> Z_c         (Z on control unchanged)
        pc = paulis[control_wire]
        pt = paulis[wire]

        # Apply independently then combine via multiplication
        # Control Pauli contribution
        new_c, new_t = _cnot_conjugate(pc, pt)
        paulis[control_wire] = new_c
        paulis[wire] = new_t

    return ''.join(paulis)


def _cnot_conjugate(pc: str, pt: str) -> tuple[str, str]:
    """Compute CNOT conjugation of Pauli pair (control, target).

    CNOT (I⊗I) CNOT = I⊗I
    CNOT (X⊗I) CNOT = X⊗X
    CNOT (I⊗X) CNOT = I⊗X
    CNOT (X⊗X) CNOT = X⊗I
    CNOT (Z⊗I) CNOT = Z⊗I
    CNOT (I⊗Z) CNOT = Z⊗Z
    CNOT (Z⊗Z) CNOT = I⊗Z
    CNOT (Y⊗I) CNOT = Y⊗X
    CNOT (I⊗Y) CNOT = Z⊗Y
    CNOT (X⊗Z) CNOT = Y⊗Y  (up to sign)
    CNOT (Z⊗X) CNOT = Z⊗X
    CNOT (Y⊗Z) CNOT = -X⊗Y  -> X⊗Y (ignoring sign)
    CNOT (Z⊗Y) CNOT = Z⊗Y... wait, this isn't right for all cases
    """
    # Use the Pauli multiplication table approach.
    # CNOT acts as: X_c -> X_c X_t, Z_t -> Z_c Z_t
    # Equivalently on the Pauli group:
    # (P_c, P_t) -> conjugated by lookup
    table = {
        ('I', 'I'): ('I', 'I'),
        ('X', 'I'): ('X', 'X'),
        ('Y', 'I'): ('Y', 'X'),
        ('Z', 'I'): ('Z', 'I'),
        ('I', 'X'): ('I', 'X'),
        ('X', 'X'): ('X', 'I'),
        ('Y', 'X'): ('Y', 'I'),
        ('Z', 'X'): ('Z', 'X'),
        ('I', 'Z'): ('Z', 'Z'),
        ('X', 'Z'): ('Y', 'Y'),
        ('Y', 'Z'): ('X', 'Y'),
        ('Z', 'Z'): ('I', 'Z'),
        ('I', 'Y'): ('Z', 'Y'),
        ('X', 'Y'): ('Y', 'Z'),
        ('Y', 'Y'): ('X', 'Z'),
        ('Z', 'Y'): ('I', 'Y'),
    }
    return table[(pc, pt)]


def _add_simple_edge(g: BaseGraph, s: VT, t: VT) -> ET:
    g.add_edge((s, t), EdgeType.SIMPLE)
    return next(g.edges(s, t))


def _add_had_edge(g: BaseGraph, s: VT, t: VT) -> ET:
    g.add_edge((s, t), EdgeType.HADAMARD)
    return next(g.edges(s, t))


def _first_neighbor(g: BaseGraph, v: VT):
    return iter(g.neighbors(v))
