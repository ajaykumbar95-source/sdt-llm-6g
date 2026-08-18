#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as ET


NS3_DIR = Path.home() / "ns-3-dev"

SOURCE = NS3_DIR / "sdt-topology.xml"
OUTPUT = NS3_DIR / "sdt-topology-visual.xml"


# ------------------------------------------------------------
# PRESENTATION-ONLY LAYOUT
#
# These coordinates affect ONLY the NetAnim presentation copy.
# They do not alter the physical positions used by ns-3/Sionna RT.
#
# Actual node IDs from the generated XML:
#   0 = gNB
#   1 = UE-1
#   2 = UE-2
#   3 = PGW
#   4 = SGW
#   5 = MME
#   6 = Remote Host
# ------------------------------------------------------------

LAYOUT = {
    # Radio access network
    1: (12.0, 10.0),     # UE-1
    2: (28.0, 10.0),     # UE-2
    0: (20.0, 5.0),      # gNB

    # 5G core
    3: (10.0, 20.0),     # PGW
    4: (20.0, 20.0),     # SGW
    5: (30.0, 20.0),     # MME

    # External network
    6: (10.0, 30.0),     # Remote Host
}


DESCRIPTIONS = {
    0: "gNB-1 | 5G-LENA + Sionna RT",
    1: "UE-1 | RNTI=1 | HEALTHY",
    2: "UE-2 | RNTI=2 | DEGRADED",
    3: "PGW | 5G Core",
    4: "SGW | 5G Core",
    5: "MME | 5G Core",
    6: "Remote Host",
}


COLORS = {
    0: (60, 120, 220),     # gNB
    1: (70, 180, 90),      # healthy UE
    2: (220, 60, 60),      # degraded UE
    3: (140, 100, 200),    # PGW
    4: (120, 120, 180),    # SGW
    5: (110, 110, 110),    # MME
    6: (150, 90, 170),     # Remote Host
}


SIZES = {
    0: (2.6, 2.6),
    1: (1.8, 1.8),
    2: (1.8, 1.8),
    3: (1.9, 1.9),
    4: (1.9, 1.9),
    5: (1.9, 1.9),
    6: (2.0, 2.0),
}


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Missing NetAnim trace: {SOURCE}"
        )

    tree = ET.parse(SOURCE)
    root = tree.getroot()

    nodes = {
        int(node.attrib["id"]): node
        for node in root.findall("node")
    }

    expected = set(LAYOUT)
    actual = set(nodes)

    missing = sorted(expected - actual)

    if missing:
        raise RuntimeError(
            f"Expected nodes missing from XML: {missing}"
        )

    # ---------------------------------------------------------
    # 1. Replace base node positions.
    # ---------------------------------------------------------

    for node_id, (x, y) in LAYOUT.items():
        nodes[node_id].set("locX", str(x))
        nodes[node_id].set("locY", str(y))

    # ---------------------------------------------------------
    # 2. Remove presentation records at t=0.
    #
    # We keep packet events and later state updates.
    # ---------------------------------------------------------

    for child in list(root):
        if child.tag != "nu":
            continue

        if child.attrib.get("t") != "0":
            continue

        if child.attrib.get("id") is None:
            continue

        if child.attrib.get("p") in {"p", "d", "s", "c"}:
            root.remove(child)

    # ---------------------------------------------------------
    # 3. Add clean presentation records.
    # ---------------------------------------------------------

    for node_id in sorted(LAYOUT):

        x, y = LAYOUT[node_id]
        r, g, b = COLORS[node_id]
        w, h = SIZES[node_id]

        ET.SubElement(
            root,
            "nu",
            {
                "p": "p",
                "t": "0",
                "id": str(node_id),
                "x": str(x),
                "y": str(y),
            },
        )

        ET.SubElement(
            root,
            "nu",
            {
                "p": "d",
                "t": "0",
                "id": str(node_id),
                "descr": DESCRIPTIONS[node_id],
            },
        )

        ET.SubElement(
            root,
            "nu",
            {
                "p": "s",
                "t": "0",
                "id": str(node_id),
                "w": str(w),
                "h": str(h),
            },
        )

        ET.SubElement(
            root,
            "nu",
            {
                "p": "c",
                "t": "0",
                "id": str(node_id),
                "r": str(r),
                "g": str(g),
                "b": str(b),
            },
        )

    # ---------------------------------------------------------
    # Add readable descriptions to ACTUAL recorded network links.
    #
    # These do not create new links. They only label links that
    # already exist in the simulation XML.
    # ---------------------------------------------------------

    link_labels = {
        (0, 4): "S1-U",
        (3, 4): "S5",
        (4, 5): "S11",
        (3, 6): "SGi",
    }

    for link in root.findall("link"):
        from_id = link.attrib.get("fromId")
        to_id = link.attrib.get("toId")

        if from_id is None or to_id is None:
            continue

        label = link_labels.get(
            (int(from_id), int(to_id))
        )

        if label:
            link.set("ld", label)

    tree.write(
        OUTPUT,
        encoding="utf-8",
        xml_declaration=False,
    )

    print("==============================================")
    print("CLEAN NETANIM TOPOLOGY GENERATED")
    print("==============================================")
    print(f"Source : {SOURCE}")
    print(f"Output : {OUTPUT}")
    print()
    print("Topology layout:")
    print("  UE-1        -> (12, 10)")
    print("  UE-2        -> (28, 10)")
    print("  gNB-1       -> (20, 5)")
    print("  PGW         -> (10, 20)")
    print("  SGW         -> (20, 20)")
    print("  MME         -> (30, 20)")
    print("  Remote Host -> (10, 30)")
    print("==============================================")


if __name__ == "__main__":
    main()
