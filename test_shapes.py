import numpy as np
import quaternion
from shapes import *


def test_pentatope_equidistance():
    vertices = pentatope()
    for v1 in vertices:
        for v2 in vertices:
            d = abs(v1 - v2)
            assert (d == 0 or np.isclose(d, np.sqrt(2.5)))


def test_orthoplex_closed():
    vertices = orthoplex(False)
    for v1 in vertices:
        for v2 in vertices:
            assert (contains(vertices, v1*v2))


def test_tesseract_symmetry():
    vertices = tesseract(False)
    symmetry = orthoplex(False)
    for v in vertices:
        for s in symmetry:
            assert (contains(vertices, v*s))


def test_tesseract_in_octaplex():
    vertices = tesseract(False)
    container = octaplex(False)
    for v1 in vertices:
        for v2 in vertices:
            assert (contains(container, v1*v2))


def test_octaplex_closed():
    vertices = octaplex(False)
    for v1 in vertices:
        for v2 in vertices:
            assert (contains(vertices, v1*v2))


def test_snub_octaplex_in_tetraplex():
    vertices = snub_octaplex(False)
    container = tetraplex(False)
    for v1 in vertices:
        for v2 in vertices:
            assert (contains(container, v1*v2))


def test_tetraplex_closed():
    vertices = tetraplex(False)
    for v1 in vertices:
        for v2 in vertices:
            assert (contains(vertices, v1*v2))


def test_generated_tetraplex():
    s = np.quaternion(0.5, 0.5, 0.5, 0.5)
    t = np.quaternion(0.5*phi, 0.5/phi, 0.5, 0)

    vertices = [s, t]

    for _ in range(4):
        verts = vertices[:]
        for v1 in verts:
            for v2 in verts:
                v = v1*v2
                if not contains(vertices, v):
                    vertices.append(v)

    assert (len(vertices) == 120)
    vertices = np.array(vertices)

    for v in tetraplex(False):
        assert (contains(vertices, v))


def test_dodecaplex():
    vertices = dodecaplex(False)
    norms = abs(vertices)
    assert (np.isclose(1, norms.min()))
    assert (np.isclose(1, norms.max()))
    assert (len(vertices) == 600)
    assert (len(dodecaplex()) == 300)


def test_dodecaplex_symmetry():
    vertices = dodecaplex(False)
    symmetry = tetraplex(False)
    for v in vertices:
        for s in symmetry:
            assert (contains(vertices, v*s))
