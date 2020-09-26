import numpy as np
import quaternion

phi = 0.5 * (1 + np.sqrt(5))


def contains(qs, q):
    if not len(qs):
        return False
    return np.isclose(0, abs(q - np.array(qs)).min())


def _permutations3(a, b, c):
    return [
        (a, b, c),
        (a, c, b),
        (b, a, c),
        (b, c, a),
        (c, a, b),
        (c, b, a)
    ]


def permutations(a, b, c, d):
    vertices = []
    for p in _permutations3(b, c, d):
        vertices.append(np.quaternion(a, *p))
    for p in _permutations3(a, c, d):
        vertices.append(np.quaternion(b, *p))
    for p in _permutations3(b, a, d):
        vertices.append(np.quaternion(c, *p))
    for p in _permutations3(b, c, a):
        vertices.append(np.quaternion(d, *p))
    return vertices


def even_permutations(a, b, c, d):
    vertices = []
    vertices.append(np.quaternion(a, b, c, d))
    vertices.append(np.quaternion(a, c, d, b))
    vertices.append(np.quaternion(a, d, b, c))

    vertices.append(np.quaternion(b, a, d, c))
    vertices.append(np.quaternion(b, d, c, a))
    vertices.append(np.quaternion(b, c, a, d))

    vertices.append(np.quaternion(c, d, a, b))
    vertices.append(np.quaternion(c, a, b, d))
    vertices.append(np.quaternion(c, b, d, a))

    vertices.append(np.quaternion(d, c, b, a))
    vertices.append(np.quaternion(d, b, a, c))
    vertices.append(np.quaternion(d, a, c, b))
    return vertices


def pentatope():
    """
    Vertices of the 5-cell
    """
    s = 3.2**-0.5
    return np.array([
        np.quaternion(1, 0, 0, 0),
        np.quaternion(-0.25, s, s, s),
        np.quaternion(-0.25, s, -s, -s),
        np.quaternion(-0.25, -s, s, -s),
        np.quaternion(-0.25, -s, -s, s),
    ])


def pentatope_v2():
    """
    Vertices of the 5-cell

    Alternative construction
    """
    vertices = np.array([
        np.quaternion(2, 0, 0, 0),
        np.quaternion(0, 2, 0, 0),
        np.quaternion(0, 0, 2, 0),
        np.quaternion(0, 0, 0, 2),
        np.quaternion(phi, phi, phi, phi),
    ])
    vertices -= np.quaternion(2+phi, 2+phi, 2+phi, 2+phi) / 5.0
    vertices /= abs(vertices)
    return vertices


def pentatope_rotors():
    """
    Symmetries of the 5-cell

    The result is two copies of the tetraplex of opposing chirality
    """
    spin = np.quaternion(0.5, 0.5, 0.5, 0.5)
    flip_left = np.quaternion(0.5, 0.5*phi, 0.5/phi, 0)
    flip_right = np.quaternion(0.5, 0.5/phi, 0.5*phi, 0)

    lefts = [spin, flip_left]
    rights = [1/spin, flip_right]

    for _ in range(4):
        ls = lefts[:]
        rs = rights[:]
        for l1, r1 in zip(ls, rs):
            for l2, r2 in zip(ls, rs):
                if not contains(lefts, l1*l2):
                    lefts.append(l1*l2)
                    rights.append(r2*r1)

    return list(zip(lefts, rights))


def orthoplex(positive=True):
    """
    Vertices of the 16-cell
    """
    if positive:
        return np.array([
            quaternion.one,
            quaternion.x,
            quaternion.y,
            quaternion.z
        ])
    else:
        return np.concatenate((orthoplex(True), -orthoplex(True)))


def tesseract(positive=True):
    """
    Vertices of the 8-cell
    """
    if positive:
        return np.array([
            np.quaternion(0.5, 0.5, 0.5, 0.5),
            np.quaternion(0.5, -0.5, 0.5, 0.5),
            np.quaternion(0.5, 0.5, -0.5, 0.5),
            np.quaternion(0.5, 0.5, 0.5, -0.5),
            np.quaternion(0.5, -0.5, -0.5, 0.5),
            np.quaternion(0.5, 0.5, -0.5, -0.5),
            np.quaternion(0.5, -0.5, 0.5, -0.5),
            np.quaternion(0.5, -0.5, -0.5, -0.5),
        ])
    else:
        return np.concatenate((tesseract(True), -tesseract(True)))


def octaplex(positive=True):
    """
    Vertices of the 24-cell
    """
    return np.concatenate((orthoplex(positive), tesseract(positive)))


def snub_octaplex(positive=True, left=True):
    """
    Vertices of a snub 24-cell
    """
    vertices = []
    for a in [0.5*phi, -0.5*phi]:
        for b in [0.5/phi, -0.5/phi]:
            for c in [0.5, -0.5]:
                if left:
                    vertices.extend(even_permutations(a, b, c, 0))
                else:
                    vertices.extend(even_permutations(b, a, c, 0))
    if positive:
        for v in vertices[:]:
            if v.w < 0:
                vertices.remove(v)
            elif v.w == 0 and v.x < 0:
                vertices.remove(v)
    return np.array(vertices)



def tetraplex(positive=True):
    """
    Vertices of the 600-cell
    """
    return np.concatenate((octaplex(positive), snub_octaplex(positive)))


def dodecaplex(positive=True):
    """
    Vertices of the 120-cell
    """
    norm = 8**0.5
    one = 1 / norm
    two = 2 / norm
    phin = phi / norm
    iphi = (phi-1) / norm
    phi2 = (phi+1) / norm
    iphi2 = 1 / (norm * (phi + 1))
    sqrt5 = np.sqrt(5) / norm

    vertices = []
    for a in (-two, two):
        for b in (-two, two):
            for q in permutations(0, 0, a, b):
                if not contains(vertices, q):
                    vertices.append(q)
    verts = []
    for a in (-one, one):
        for b in (-one, one):
            for c in (-one, one):
                for d in (-sqrt5, sqrt5):
                    for q in permutations(a, b, c, d):
                        if not contains(verts, q):
                            verts.append(q)
    vertices.extend(verts)
    verts = []
    for a in (-iphi2, iphi2):
        for b in (-phin, phin):
            for c in (-phin, phin):
                for d in (-phin, phin):
                    for q in permutations(a, b, c, d):
                        if not contains(verts, q):
                            verts.append(q)
    vertices.extend(verts)
    verts = []
    for a in (-iphi, iphi):
        for b in (-iphi, iphi):
            for c in (-iphi, iphi):
                for d in (-phi2, phi2):
                    for q in permutations(a, b, c, d):
                        if not contains(verts, q):
                            verts.append(q)
    vertices.extend(verts)
    for a in (-iphi2, iphi2):
        for b in (-one, one):
            for c in (-phi2, phi2):
                vertices.extend(even_permutations(0, a, b, c))
    for a in (-iphi, iphi):
        for b in (-phin, phin):
            for c in (-sqrt5, sqrt5):
                vertices.extend(even_permutations(0, a, b, c))
    for a in (-iphi, iphi):
        for b in (-one, one):
            for c in (-phin, phin):
                for d in (-two, two):
                    vertices.extend(even_permutations(a, b, c, d))

    if positive:
        for v in vertices[:]:
            if v.w < 0:
                vertices.remove(v)
            elif v.w == 0 and v.x < 0:
                vertices.remove(v)
            elif v.w == 0 and v.x == 0 and v.y < 0:
                vertices.remove(v)

    return np.array(vertices)
