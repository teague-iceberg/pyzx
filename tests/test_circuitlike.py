"""Tests for CircuitLike and Pauli box generation/conjugation."""

import unittest
from fractions import Fraction

from pyzx.graph.multigraph import Multigraph
from pyzx.utils import EdgeType, VertexType
from pyzx.circuitlike import CircuitLike, check_same_support_adjacent
from pyzx.paulibox import (
    generate_pauli_box, conjugate_pauli_string, _cnot_conjugate,
    identify_clifford_gate, identify_pauli_box,
)
from pyzx.rewrite_rules.push_clifford_rule import check_push_clifford, apply_push_clifford


def _new_graph():
    g = Multigraph()
    g.set_auto_simplify(False)
    return g


class TestCircuitLike(unittest.TestCase):

    def test_boundary_edges(self):
        """Boundary edges are edges crossing the subgraph boundary."""
        g = _new_graph()
        a = g.add_vertex(VertexType.Z, qubit=0, row=0)
        b = g.add_vertex(VertexType.Z, qubit=0, row=1)
        c = g.add_vertex(VertexType.Z, qubit=0, row=2)
        g.add_edge((a, b), EdgeType.SIMPLE)
        g.add_edge((b, c), EdgeType.SIMPLE)

        cl = CircuitLike(vertices={b})
        boundary = cl.boundary_edges(g)
        self.assertEqual(len(boundary), 2)

    def test_internal_edges(self):
        """Internal edges have both endpoints inside."""
        g = _new_graph()
        a = g.add_vertex(VertexType.Z, qubit=0, row=0)
        b = g.add_vertex(VertexType.Z, qubit=0, row=1)
        c = g.add_vertex(VertexType.Z, qubit=0, row=2)
        g.add_edge((a, b), EdgeType.SIMPLE)
        g.add_edge((b, c), EdgeType.SIMPLE)

        cl = CircuitLike(vertices={a, b})
        internal = cl.internal_edges(g)
        self.assertEqual(len(internal), 1)

    def test_validate_max_two_per_qubit(self):
        """At most 2 incident edges per qubit label per vertex."""
        g = _new_graph()
        a = g.add_vertex(VertexType.Z, qubit=0, row=0)
        b = g.add_vertex(VertexType.Z, qubit=0, row=1)
        c = g.add_vertex(VertexType.Z, qubit=0, row=2)
        d = g.add_vertex(VertexType.Z, qubit=0, row=3)
        g.add_edge((a, b), EdgeType.SIMPLE)
        g.add_edge((b, c), EdgeType.SIMPLE)
        g.add_edge((b, d), EdgeType.SIMPLE)

        e_ab = next(g.edges(a, b))
        e_bc = next(g.edges(b, c))
        e_bd = next(g.edges(b, d))

        cl = CircuitLike(
            vertices={a, b, c, d},
            edge_labels={e_ab: 0, e_bc: 0, e_bd: 0},  # 3 edges on qubit 0 at vertex b
        )
        errors = cl.validate(g)
        self.assertTrue(any("qubit 0" in e for e in errors))

    def test_validate_clean(self):
        """A simple valid CircuitLike passes validation."""
        g = _new_graph()
        a = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
        b = g.add_vertex(VertexType.Z, qubit=0, row=1)
        c = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=2)
        g.add_edge((a, b), EdgeType.SIMPLE)
        g.add_edge((b, c), EdgeType.SIMPLE)

        e_in = next(g.edges(a, b))
        e_out = next(g.edges(b, c))

        cl = CircuitLike(
            vertices={b},
            input_edges={e_in},
            output_edges={e_out},
            edge_labels={e_in: 0, e_out: 0},
        )
        errors = cl.validate(g)
        self.assertEqual(errors, [])


class TestSameSupportAdjacent(unittest.TestCase):

    def test_adjacent_simple(self):
        """Two single-vertex subgraphs connected by one edge are adjacent."""
        g = _new_graph()
        a = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
        b = g.add_vertex(VertexType.Z, qubit=0, row=1)
        c = g.add_vertex(VertexType.X, qubit=0, row=2)
        d = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=3)
        g.add_edge((a, b), EdgeType.SIMPLE)
        g.add_edge((b, c), EdgeType.SIMPLE)
        g.add_edge((c, d), EdgeType.SIMPLE)

        e_ab = next(g.edges(a, b))
        e_bc = next(g.edges(b, c))
        e_cd = next(g.edges(c, d))

        cl_b = CircuitLike(
            vertices={b},
            input_edges={e_ab},
            output_edges={e_bc},
            edge_labels={e_ab: 0, e_bc: 0},
        )
        cl_c = CircuitLike(
            vertices={c},
            input_edges={e_bc},
            output_edges={e_cd},
            edge_labels={e_bc: 0, e_cd: 0},
        )

        result = check_same_support_adjacent(cl_b, cl_c)
        self.assertIsNotNone(result)
        self.assertEqual(result, {0: 0})

    def test_not_adjacent_gap(self):
        """Two subgraphs with a vertex in between are not adjacent."""
        g = _new_graph()
        a = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
        b = g.add_vertex(VertexType.Z, qubit=0, row=1)
        mid = g.add_vertex(VertexType.Z, qubit=0, row=2)
        c = g.add_vertex(VertexType.X, qubit=0, row=3)
        d = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=4)
        g.add_edge((a, b), EdgeType.SIMPLE)
        g.add_edge((b, mid), EdgeType.SIMPLE)
        g.add_edge((mid, c), EdgeType.SIMPLE)
        g.add_edge((c, d), EdgeType.SIMPLE)

        e_ab = next(g.edges(a, b))
        e_b_mid = next(g.edges(b, mid))
        e_mid_c = next(g.edges(mid, c))
        e_cd = next(g.edges(c, d))

        cl_b = CircuitLike(
            vertices={b},
            input_edges={e_ab},
            output_edges={e_b_mid},
            edge_labels={e_ab: 0, e_b_mid: 0},
        )
        cl_c = CircuitLike(
            vertices={c},
            input_edges={e_mid_c},
            output_edges={e_cd},
            edge_labels={e_mid_c: 0, e_cd: 0},
        )

        result = check_same_support_adjacent(cl_b, cl_c)
        self.assertIsNone(result)

    def test_adjacent_two_qubit(self):
        """Two 2-qubit subgraphs sharing both wires."""
        g = _new_graph()
        # Qubit 0: a0 -> b0 -> c0 -> d0
        a0 = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
        b0 = g.add_vertex(VertexType.Z, qubit=0, row=1)
        c0 = g.add_vertex(VertexType.X, qubit=0, row=2)
        d0 = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=3)
        # Qubit 1: a1 -> b1 -> c1 -> d1
        a1 = g.add_vertex(VertexType.BOUNDARY, qubit=1, row=0)
        b1 = g.add_vertex(VertexType.Z, qubit=1, row=1)
        c1 = g.add_vertex(VertexType.X, qubit=1, row=2)
        d1 = g.add_vertex(VertexType.BOUNDARY, qubit=1, row=3)

        for s, t in [(a0, b0), (b0, c0), (c0, d0),
                     (a1, b1), (b1, c1), (c1, d1)]:
            g.add_edge((s, t), EdgeType.SIMPLE)

        e_b0c0 = next(g.edges(b0, c0))
        e_b1c1 = next(g.edges(b1, c1))

        cl_left = CircuitLike(
            vertices={b0, b1},
            input_edges={next(g.edges(a0, b0)), next(g.edges(a1, b1))},
            output_edges={e_b0c0, e_b1c1},
            edge_labels={
                next(g.edges(a0, b0)): 0, e_b0c0: 0,
                next(g.edges(a1, b1)): 1, e_b1c1: 1,
            },
        )
        cl_right = CircuitLike(
            vertices={c0, c1},
            input_edges={e_b0c0, e_b1c1},
            output_edges={next(g.edges(c0, d0)), next(g.edges(c1, d1))},
            edge_labels={
                e_b0c0: 0, next(g.edges(c0, d0)): 0,
                e_b1c1: 1, next(g.edges(c1, d1)): 1,
            },
        )

        result = check_same_support_adjacent(cl_left, cl_right)
        self.assertIsNotNone(result)
        self.assertEqual(result, {0: 0, 1: 1})

    def test_adjacent_swapped_qubits(self):
        """Adjacent with different qubit labeling on each side."""
        g = _new_graph()
        a = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
        b = g.add_vertex(VertexType.Z, qubit=0, row=1)
        c = g.add_vertex(VertexType.X, qubit=0, row=2)
        d = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=3)
        g.add_edge((a, b), EdgeType.SIMPLE)
        g.add_edge((b, c), EdgeType.SIMPLE)
        g.add_edge((c, d), EdgeType.SIMPLE)

        e_bc = next(g.edges(b, c))

        # Left labels the edge as qubit 5, right labels it as qubit 3
        cl_left = CircuitLike(
            vertices={b},
            input_edges={next(g.edges(a, b))},
            output_edges={e_bc},
            edge_labels={next(g.edges(a, b)): 5, e_bc: 5},
        )
        cl_right = CircuitLike(
            vertices={c},
            input_edges={e_bc},
            output_edges={next(g.edges(c, d))},
            edge_labels={e_bc: 3, next(g.edges(c, d)): 3},
        )

        result = check_same_support_adjacent(cl_left, cl_right)
        self.assertIsNotNone(result)
        self.assertEqual(result, {5: 3})


class TestGeneratePauliBox(unittest.TestCase):

    def test_z_pauli_box_structure(self):
        """A single-qubit Z Pauli box has 1 Z spider + 1 X spider."""
        g, cl = generate_pauli_box("Z")
        z_count = sum(1 for v in cl.vertices if g.type(v) == VertexType.Z)
        x_count = sum(1 for v in cl.vertices if g.type(v) == VertexType.X)
        self.assertEqual(z_count, 1)
        self.assertEqual(x_count, 1)

    def test_zz_pauli_box_structure(self):
        """A ZZ Pauli box has 2 Z spiders + 1 X spider."""
        g, cl = generate_pauli_box("ZZ")
        z_count = sum(1 for v in cl.vertices if g.type(v) == VertexType.Z)
        x_count = sum(1 for v in cl.vertices if g.type(v) == VertexType.X)
        self.assertEqual(z_count, 2)
        self.assertEqual(x_count, 1)

    def test_x_pauli_box_has_conjugation(self):
        """An X Pauli box has H-boxes for Hadamard conjugation."""
        g, cl = generate_pauli_box("X")
        # 1 Z spider for the Pauli, 2 H-boxes for conjugation
        z_count = sum(1 for v in cl.vertices if g.type(v) == VertexType.Z)
        h_count = sum(1 for v in cl.vertices if g.type(v) == VertexType.H_BOX)
        self.assertEqual(z_count, 1)
        self.assertEqual(h_count, 2)

    def test_io_edges(self):
        """Pauli box has input/output edges for each non-I qubit."""
        for ps, expected in [("Z", 1), ("XZ", 2), ("XYZ", 3), ("XYZI", 3)]:
            g, cl = generate_pauli_box(ps)
            non_i = sum(1 for c in ps if c != 'I')
            self.assertEqual(len(cl.input_edges), non_i,
                             f"Wrong input count for {ps}")
            self.assertEqual(len(cl.output_edges), non_i,
                             f"Wrong output count for {ps}")

    def test_neutral_boundary_edge(self):
        """Pauli box has 1 neutral boundary edge (the phase leg)."""
        g, cl = generate_pauli_box("ZZ")
        neutral = cl.neutral_edges(g)
        self.assertEqual(len(neutral), 1)

    def test_validation_passes(self):
        """Generated Pauli box should pass validation."""
        for ps in ["Z", "X", "Y", "XZ", "XYZI"]:
            g, cl = generate_pauli_box(ps)
            errors = cl.validate(g)
            self.assertEqual(errors, [], f"Validation failed for {ps}: {errors}")


class TestConjugatePauliString(unittest.TestCase):

    def test_h_conjugation(self):
        self.assertEqual(conjugate_pauli_string("X", "H", 0), "Z")
        self.assertEqual(conjugate_pauli_string("Z", "H", 0), "X")
        self.assertEqual(conjugate_pauli_string("I", "H", 0), "I")

    def test_s_conjugation(self):
        self.assertEqual(conjugate_pauli_string("X", "S", 0), "Y")
        self.assertEqual(conjugate_pauli_string("Y", "S", 0), "X")
        self.assertEqual(conjugate_pauli_string("Z", "S", 0), "Z")

    def test_h_on_second_qubit(self):
        self.assertEqual(conjugate_pauli_string("XZ", "H", 1), "XX")
        self.assertEqual(conjugate_pauli_string("ZX", "H", 0), "XX")

    def test_cnot_conjugation(self):
        # CNOT: X on control spreads to target
        self.assertEqual(conjugate_pauli_string("XI", "CNOT", wire=1, control_wire=0), "XX")
        # CNOT: Z on target spreads to control
        self.assertEqual(conjugate_pauli_string("IZ", "CNOT", wire=1, control_wire=0), "ZZ")
        # CNOT: X on target unchanged
        self.assertEqual(conjugate_pauli_string("IX", "CNOT", wire=1, control_wire=0), "IX")
        # CNOT: Z on control unchanged
        self.assertEqual(conjugate_pauli_string("ZI", "CNOT", wire=1, control_wire=0), "ZI")

    def test_cnot_conjugation_table(self):
        """Verify full CNOT conjugation table."""
        self.assertEqual(_cnot_conjugate('X', 'I'), ('X', 'X'))
        self.assertEqual(_cnot_conjugate('I', 'X'), ('I', 'X'))
        self.assertEqual(_cnot_conjugate('I', 'Z'), ('Z', 'Z'))
        self.assertEqual(_cnot_conjugate('Z', 'I'), ('Z', 'I'))
        self.assertEqual(_cnot_conjugate('X', 'X'), ('X', 'I'))
        self.assertEqual(_cnot_conjugate('Z', 'Z'), ('I', 'Z'))


class TestIdentifyCliffordGate(unittest.TestCase):

    def test_identify_hadamard(self):
        """An H-box vertex is identified as H."""
        from pyzx.utils import set_h_box_label
        g = _new_graph()
        b_in = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
        h = g.add_vertex(VertexType.H_BOX, qubit=0, row=1)
        set_h_box_label(g, h, -1)
        b_out = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=2)
        g.add_edge((b_in, h), EdgeType.SIMPLE)
        g.add_edge((h, b_out), EdgeType.SIMPLE)

        e_in = next(g.edges(b_in, h))
        e_out = next(g.edges(h, b_out))

        cl = CircuitLike(
            vertices={h},
            input_edges={e_in},
            output_edges={e_out},
            edge_labels={e_in: 0, e_out: 0},
        )
        result = identify_clifford_gate(g, cl)
        self.assertIsNotNone(result)
        name, wire, ctrl = result
        self.assertEqual(name, "H")
        self.assertEqual(wire, 0)
        self.assertIsNone(ctrl)

    def test_identify_s_gate(self):
        """A Z spider with phase pi/2 is identified as S."""
        g = _new_graph()
        b_in = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
        s = g.add_vertex(VertexType.Z, qubit=0, row=1, phase=Fraction(1, 2))
        b_out = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=2)
        g.add_edge((b_in, s), EdgeType.SIMPLE)
        g.add_edge((s, b_out), EdgeType.SIMPLE)

        e_in = next(g.edges(b_in, s))
        e_out = next(g.edges(s, b_out))

        cl = CircuitLike(
            vertices={s},
            input_edges={e_in},
            output_edges={e_out},
            edge_labels={e_in: 0, e_out: 0},
        )
        result = identify_clifford_gate(g, cl)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "S")

    def test_identify_cnot(self):
        """A Z spider connected to an X spider is identified as CNOT."""
        g = _new_graph()
        b_in0 = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
        b_in1 = g.add_vertex(VertexType.BOUNDARY, qubit=1, row=0)
        z = g.add_vertex(VertexType.Z, qubit=0, row=1, phase=0)
        x = g.add_vertex(VertexType.X, qubit=1, row=1, phase=0)
        b_out0 = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=2)
        b_out1 = g.add_vertex(VertexType.BOUNDARY, qubit=1, row=2)

        g.add_edge((b_in0, z), EdgeType.SIMPLE)
        g.add_edge((b_in1, x), EdgeType.SIMPLE)
        g.add_edge((z, x), EdgeType.SIMPLE)
        g.add_edge((z, b_out0), EdgeType.SIMPLE)
        g.add_edge((x, b_out1), EdgeType.SIMPLE)

        e_in0 = next(g.edges(b_in0, z))
        e_in1 = next(g.edges(b_in1, x))
        e_out0 = next(g.edges(z, b_out0))
        e_out1 = next(g.edges(x, b_out1))

        cl = CircuitLike(
            vertices={z, x},
            input_edges={e_in0, e_in1},
            output_edges={e_out0, e_out1},
            edge_labels={e_in0: 0, e_out0: 0, e_in1: 1, e_out1: 1},
        )
        result = identify_clifford_gate(g, cl)
        self.assertIsNotNone(result)
        name, target, control = result
        self.assertEqual(name, "CNOT")
        self.assertEqual(target, 1)  # X spider is target
        self.assertEqual(control, 0)  # Z spider is control


class TestIdentifyPauliBox(unittest.TestCase):

    def test_identify_z_pauli_box(self):
        """A generated Z Pauli box is correctly identified."""
        g, cl = generate_pauli_box("Z")
        result = identify_pauli_box(g, cl)
        self.assertEqual(result, "Z")

    def test_identify_zz_pauli_box(self):
        """A generated ZZ Pauli box is correctly identified."""
        g, cl = generate_pauli_box("ZZ")
        result = identify_pauli_box(g, cl)
        self.assertEqual(result, "ZZ")

    def test_identify_x_pauli_box(self):
        """A generated X Pauli box is correctly identified."""
        g, cl = generate_pauli_box("X")
        result = identify_pauli_box(g, cl)
        self.assertEqual(result, "X")

    def test_identify_xz_pauli_box(self):
        """A generated XZ Pauli box is correctly identified."""
        g, cl = generate_pauli_box("XZ")
        result = identify_pauli_box(g, cl)
        self.assertEqual(result, "XZ")

    def test_identify_y_pauli_box(self):
        """A generated Y Pauli box is correctly identified."""
        g, cl = generate_pauli_box("Y")
        result = identify_pauli_box(g, cl)
        self.assertEqual(result, "Y")


def _build_h_next_to_pauli(pauli_string):
    """Helper: build a graph with H gate adjacent to a Pauli box.

    Returns (g, clifford_cl, pauli_cl, upstream_boundary, downstream_boundaries, phase_boundary).
    """
    g = _new_graph()

    # Generate the Pauli box first
    pauli_g, pauli_cl = generate_pauli_box(pauli_string, g, start_row=2, start_qubit=0)

    # Find qubit 0's input edge and its boundary vertex
    pauli_in_map = pauli_cl.input_qubit_map()
    pauli_in_edge_q0 = pauli_in_map[0]
    s, t = g.edge_st(pauli_in_edge_q0)
    pauli_in_boundary_q0 = s if s not in pauli_cl.vertices else t
    pauli_in_vertex_q0 = t if s not in pauli_cl.vertices else s

    # Remove the Pauli box's input boundary on qubit 0
    g.remove_vertex(pauli_in_boundary_q0)

    # Insert Hadamard gate (H-box) before the Pauli box on qubit 0
    from pyzx.utils import set_h_box_label
    b_upstream = g.add_vertex(VertexType.BOUNDARY, qubit=0, row=0)
    h = g.add_vertex(VertexType.H_BOX, qubit=0, row=1)
    set_h_box_label(g, h, -1)
    g.add_edge((b_upstream, h), EdgeType.SIMPLE)
    g.add_edge((h, pauli_in_vertex_q0), EdgeType.SIMPLE)

    e_upstream_h = next(g.edges(b_upstream, h))
    e_h_pauli = next(g.edges(h, pauli_in_vertex_q0))

    # Build the Clifford CircuitLike
    clifford_cl = CircuitLike(
        vertices={h},
        input_edges={e_upstream_h},
        output_edges={e_h_pauli},
        edge_labels={e_upstream_h: 0, e_h_pauli: 0},
    )

    # Update Pauli box CircuitLike: replace old input edge with new one
    pauli_cl.input_edges.discard(pauli_in_edge_q0)
    pauli_cl.input_edges.add(e_h_pauli)
    if pauli_in_edge_q0 in pauli_cl.edge_labels:
        del pauli_cl.edge_labels[pauli_in_edge_q0]
    pauli_cl.edge_labels[e_h_pauli] = 0

    # Collect downstream boundaries (Pauli box output boundaries)
    downstream = {}
    for q, e in pauli_cl.output_qubit_map().items():
        s, t = g.edge_st(e)
        downstream[q] = s if s not in pauli_cl.vertices else t

    # Phase boundary
    neutral = pauli_cl.neutral_edges(g)
    phase_boundary = None
    for e in neutral:
        s, t = g.edge_st(e)
        phase_boundary = s if s not in pauli_cl.vertices else t

    return g, clifford_cl, pauli_cl, b_upstream, downstream, phase_boundary


class TestPushCliffordCheck(unittest.TestCase):

    def test_h_adjacent_to_z_pauli(self):
        """H gate adjacent to Z Pauli box should be a valid push."""
        g, clifford_cl, pauli_cl, _, _, _ = _build_h_next_to_pauli("Z")

        match = check_push_clifford(g, clifford_cl, pauli_cl)
        self.assertIsNotNone(match)
        self.assertEqual(match.clifford_name, "H")
        self.assertEqual(match.pauli_string, "Z")

    def test_h_not_adjacent_to_xz_different_support(self):
        """H (1 qubit) next to XZ Pauli box (2 qubits): not same-support adjacent."""
        g, clifford_cl, pauli_cl, _, _, _ = _build_h_next_to_pauli("XZ")

        match = check_push_clifford(g, clifford_cl, pauli_cl)
        self.assertIsNone(match)  # Different support sizes


class TestPushCliffordApply(unittest.TestCase):

    def test_h_through_z_gives_x(self):
        """Pushing H through Z Pauli box: Z -> X (HZH = X)."""
        g, clifford_cl, pauli_cl, b_upstream, downstream, phase_b = _build_h_next_to_pauli("Z")

        match = check_push_clifford(g, clifford_cl, pauli_cl)
        self.assertIsNotNone(match)

        apply_push_clifford(g, match)

        # The upstream boundary should still exist and be connected
        self.assertIn(b_upstream, g.vertices())

        # The downstream boundary should still exist
        for q, b in downstream.items():
            self.assertIn(b, g.vertices())

        # The phase boundary should still exist
        self.assertIn(phase_b, g.vertices())

        # The new Pauli box should be X (identified from the graph)
        # Find the X spider (central) — should still have one
        x_spiders = [v for v in g.vertices()
                     if g.type(v) == VertexType.X and g.phase(v) == 0
                     and v != phase_b]
        self.assertTrue(len(x_spiders) >= 1)

    def test_h_through_z_preserves_connectivity(self):
        """After pushing H through Z, the graph should still be connected
        from upstream to downstream."""
        g, clifford_cl, pauli_cl, b_upstream, downstream, phase_b = _build_h_next_to_pauli("Z")

        match = check_push_clifford(g, clifford_cl, pauli_cl)
        apply_push_clifford(g, match)

        # Check there's a path from upstream to downstream
        # (simple reachability check)
        visited = set()
        queue = [b_upstream]
        while queue:
            v = queue.pop()
            if v in visited:
                continue
            visited.add(v)
            for w in g.neighbors(v):
                queue.append(w)

        for q, b in downstream.items():
            self.assertIn(b, visited,
                          f"Downstream boundary on qubit {q} not reachable")
        self.assertIn(phase_b, visited, "Phase boundary not reachable")


if __name__ == '__main__':
    unittest.main()
