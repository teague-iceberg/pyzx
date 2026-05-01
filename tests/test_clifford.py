import unittest
from fractions import Fraction

from pyzx import Circuit
from pyzx.clifford import graph_is_in_clifford_fragment, graph_is_unitary_in_clifford_fragment
from pyzx.generate import spider
from pyzx.graph import Graph
from pyzx.symbolic import new_const, new_var
from pyzx.utils import EdgeType, VertexType, set_h_box_label


def only_non_boundary_vertex(g):
    vertices = [v for v in g.vertices() if g.type(v) != VertexType.BOUNDARY]
    assert len(vertices) == 1
    return vertices[0]


def one_node_graph(vertex_type, phase):
    g = Graph()
    i = g.add_vertex(VertexType.BOUNDARY, 0, 0)
    v = g.add_vertex(vertex_type, 0, 1, phase)
    o = g.add_vertex(VertexType.BOUNDARY, 0, 2)
    g.add_edges([(i, v), (v, o)])
    g.set_inputs((i,))
    g.set_outputs((o,))
    return g


def one_leg_node_graph(vertex_type, phase):
    g = Graph()
    v = g.add_vertex(vertex_type, 0, 1, phase)
    o = g.add_vertex(VertexType.BOUNDARY, 0, 2)
    g.add_edge((v, o))
    g.set_outputs((o,))
    return g


def zero_leg_node_graph(vertex_type, phase):
    g = Graph()
    g.add_vertex(vertex_type, 0, 0, phase)
    return g


def manual_w_node_graph(outputs):
    g = Graph()
    w_in = g.add_vertex(VertexType.W_INPUT, 0, 0)
    w_out = g.add_vertex(VertexType.W_OUTPUT, 0, 1)
    g.add_edge((w_in, w_out), EdgeType.W_IO)
    out = []
    for i in range(outputs):
        b = g.add_vertex(VertexType.BOUNDARY, i, 2)
        out.append(b)
        g.add_edge((w_out, b))
    g.set_outputs(tuple(out))
    return g


def unpaired_w_node_graph():
    g = Graph()
    w = g.add_vertex(VertexType.W_INPUT, 0, 0)
    b = g.add_vertex(VertexType.BOUNDARY, 0, 1)
    g.add_edge((w, b))
    g.set_outputs((b,))
    return g


class TestCliffordFragment(unittest.TestCase):
    def test_z_and_x_spider_phases(self):
        self.assertTrue(graph_is_in_clifford_fragment(spider("Z", 1, 1, Fraction(1, 2))))
        self.assertTrue(graph_is_in_clifford_fragment(spider("X", 1, 1, 1)))
        self.assertFalse(graph_is_in_clifford_fragment(spider("Z", 1, 1, Fraction(1, 4))))

    def test_symbolic_z_spider_phases(self):
        bool_var = new_var("b", True)
        real_var = new_var("x", False)
        self.assertTrue(graph_is_in_clifford_fragment(one_node_graph(VertexType.Z, bool_var / 2)))
        self.assertFalse(graph_is_in_clifford_fragment(one_node_graph(VertexType.Z, real_var / 2)))
        self.assertFalse(graph_is_in_clifford_fragment(one_node_graph(VertexType.Z, bool_var / 4)))

    def test_z_box_labels(self):
        g = spider("ZBox", 1, 1, 1j)
        self.assertTrue(graph_is_in_clifford_fragment(g))

        g = spider("ZBox", 1, 1, 0)
        self.assertTrue(graph_is_in_clifford_fragment(g))

        g = spider("ZBox", 0, 0, -1j)
        self.assertTrue(graph_is_in_clifford_fragment(g))

        g = spider("ZBox", 1, 1, 2)
        self.assertFalse(graph_is_in_clifford_fragment(g))

        g = spider("ZBox", 0, 0, 2)
        self.assertFalse(graph_is_in_clifford_fragment(g))

    def test_symbolic_z_box_constant_labels(self):
        self.assertTrue(graph_is_in_clifford_fragment(spider("ZBox", 1, 1, new_const(0))))
        self.assertTrue(graph_is_in_clifford_fragment(spider("ZBox", 1, 1, new_const(1))))
        self.assertTrue(graph_is_in_clifford_fragment(spider("ZBox", 1, 1, new_const(1j))))
        self.assertTrue(graph_is_in_clifford_fragment(spider("ZBox", 1, 1, new_const(-1))))
        self.assertTrue(graph_is_in_clifford_fragment(spider("ZBox", 1, 1, new_const(-1j))))
        self.assertFalse(graph_is_in_clifford_fragment(spider("ZBox", 1, 1, new_var("a", False))))

    def test_h_box_labels_by_arity(self):
        zero_leg = spider("H", 0, 0, 0)
        set_h_box_label(zero_leg, only_non_boundary_vertex(zero_leg), 0)
        self.assertTrue(graph_is_in_clifford_fragment(zero_leg))

        zero_leg = spider("H", 0, 0, 0)
        set_h_box_label(zero_leg, only_non_boundary_vertex(zero_leg), (1 + 1j) / 2**0.5)
        self.assertTrue(graph_is_in_clifford_fragment(zero_leg))

        zero_leg = spider("H", 0, 0, 0)
        set_h_box_label(zero_leg, only_non_boundary_vertex(zero_leg), 2)
        self.assertFalse(graph_is_in_clifford_fragment(zero_leg))

        one_leg = spider("H", 1, 0, 0)
        set_h_box_label(one_leg, only_non_boundary_vertex(one_leg), 0)
        self.assertTrue(graph_is_in_clifford_fragment(one_leg))

        for label in (1, 1j, -1, -1j):
            one_leg = spider("H", 1, 0, 0)
            set_h_box_label(one_leg, only_non_boundary_vertex(one_leg), label)
            self.assertTrue(graph_is_in_clifford_fragment(one_leg))

        two_leg_zero = spider("H", 1, 1, 0)
        set_h_box_label(two_leg_zero, only_non_boundary_vertex(two_leg_zero), 0)
        self.assertFalse(graph_is_in_clifford_fragment(two_leg_zero))

        two_leg_identity = spider("H", 1, 1, 0)
        set_h_box_label(two_leg_identity, only_non_boundary_vertex(two_leg_identity), 1)
        self.assertTrue(graph_is_in_clifford_fragment(two_leg_identity))

        two_leg_hadamard = spider("H", 1, 1, 0)
        set_h_box_label(two_leg_hadamard, only_non_boundary_vertex(two_leg_hadamard), -1)
        self.assertTrue(graph_is_in_clifford_fragment(two_leg_hadamard))

        three_leg_standard_hbox = spider("H", 1, 2, 0)
        set_h_box_label(three_leg_standard_hbox, only_non_boundary_vertex(three_leg_standard_hbox), -1)
        self.assertFalse(graph_is_in_clifford_fragment(three_leg_standard_hbox))

        three_leg_all_ones_hbox = spider("H", 1, 2, 0)
        set_h_box_label(three_leg_all_ones_hbox, only_non_boundary_vertex(three_leg_all_ones_hbox), 1)
        self.assertTrue(graph_is_in_clifford_fragment(three_leg_all_ones_hbox))

    def test_symbolic_h_box_labels(self):
        one_leg = spider("H", 1, 0, 0)
        one_leg.set_vdata(only_non_boundary_vertex(one_leg), "label", new_const(0))
        self.assertTrue(graph_is_in_clifford_fragment(one_leg))

        two_leg = spider("H", 1, 1, 0)
        two_leg.set_vdata(only_non_boundary_vertex(two_leg), "label", new_const(-1))
        self.assertTrue(graph_is_in_clifford_fragment(two_leg))

        three_leg = spider("H", 1, 2, 0)
        three_leg.set_vdata(only_non_boundary_vertex(three_leg), "label", new_const(-1))
        self.assertFalse(graph_is_in_clifford_fragment(three_leg))

        symbolic_label = spider("H", 1, 1, 0)
        symbolic_label.set_vdata(only_non_boundary_vertex(symbolic_label), "label", new_var("a", False))
        self.assertFalse(graph_is_in_clifford_fragment(symbolic_label))

    def test_symbolic_h_box_legacy_phases(self):
        bool_var = new_var("b", True)
        real_var = new_var("x", False)

        two_leg_hadamard = one_node_graph(VertexType.H_BOX, new_const(1))
        self.assertTrue(graph_is_in_clifford_fragment(two_leg_hadamard))

        zero_leg_quarter_phase = zero_leg_node_graph(VertexType.H_BOX, bool_var / 4)
        self.assertTrue(graph_is_in_clifford_fragment(zero_leg_quarter_phase))

        zero_leg_eighth_phase = zero_leg_node_graph(VertexType.H_BOX, bool_var / 8)
        self.assertFalse(graph_is_in_clifford_fragment(zero_leg_eighth_phase))

        two_leg_non_clifford = one_node_graph(VertexType.H_BOX, new_const(Fraction(1, 2)))
        self.assertFalse(graph_is_in_clifford_fragment(two_leg_non_clifford))

        one_leg_bool_half_integer_phase = one_leg_node_graph(VertexType.H_BOX, bool_var / 2)
        self.assertTrue(graph_is_in_clifford_fragment(one_leg_bool_half_integer_phase))

        two_leg_bool_integer_phase = one_node_graph(VertexType.H_BOX, bool_var)
        self.assertTrue(graph_is_in_clifford_fragment(two_leg_bool_integer_phase))

        two_leg_real_integer_phase = one_node_graph(VertexType.H_BOX, real_var)
        self.assertFalse(graph_is_in_clifford_fragment(two_leg_real_integer_phase))

        three_leg_hadamard = spider("H", 1, 2, 0)
        h = only_non_boundary_vertex(three_leg_hadamard)
        three_leg_hadamard.set_phase(h, new_const(1))
        self.assertFalse(graph_is_in_clifford_fragment(three_leg_hadamard))

        three_leg_even_bool_phase = spider("H", 1, 2, 0)
        h = only_non_boundary_vertex(three_leg_even_bool_phase)
        three_leg_even_bool_phase.set_phase(h, 2 * bool_var)
        self.assertTrue(graph_is_in_clifford_fragment(three_leg_even_bool_phase))

        three_leg_odd_bool_phase = spider("H", 1, 2, 0)
        h = only_non_boundary_vertex(three_leg_odd_bool_phase)
        three_leg_odd_bool_phase.set_phase(h, bool_var)
        self.assertFalse(graph_is_in_clifford_fragment(three_leg_odd_bool_phase))

    def test_w_node_arity(self):
        self.assertTrue(graph_is_in_clifford_fragment(spider("W", 1, 1)))
        self.assertFalse(graph_is_in_clifford_fragment(spider("W", 1, 2)))
        self.assertTrue(graph_is_in_clifford_fragment(manual_w_node_graph(2)))
        self.assertFalse(graph_is_in_clifford_fragment(manual_w_node_graph(3)))
        self.assertFalse(graph_is_in_clifford_fragment(unpaired_w_node_graph()))


class TestUnitaryInCliffordFragment(unittest.TestCase):
    def test_clifford_fragment_unitary_circuit_graph(self):
        c = Circuit(2)
        c.add_gate("HAD", 0)
        c.add_gate("CNOT", 0, 1)
        self.assertTrue(graph_is_unitary_in_clifford_fragment(c.to_graph()))

    def test_unitary_non_clifford_fragment_graph(self):
        c = Circuit(1)
        c.add_gate("T", 0)
        self.assertFalse(graph_is_unitary_in_clifford_fragment(c.to_graph()))

    def test_non_unitary_graph(self):
        self.assertFalse(graph_is_unitary_in_clifford_fragment(spider("Z", 0, 1, 0)))

    def test_requires_square_map(self):
        self.assertFalse(graph_is_unitary_in_clifford_fragment(spider("Z", 1, 2, 0)))


if __name__ == "__main__":
    unittest.main()
