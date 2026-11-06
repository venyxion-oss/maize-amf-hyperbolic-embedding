import os
import gc
import csv
import warnings
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.sparse.csgraph import dijkstra
import networkx as nx
import xml.etree.ElementTree as ET

import torch
import torch.nn.functional as F
from sklearn.manifold import MDS, SpectralEmbedding


torch.set_num_threads(8)
warnings.filterwarnings("ignore")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_MAX_NODES = 400

SEEDS = [
    42,
    123,
    456,
    789,
    1011
]

EMBEDDING_DIMENSIONS = [
    2,
    16,
    32
]

MODELS = [
    "Poincare",
    "MDS",
    "Spectral"
]

ENVIRONMENTAL_CONDITIONS = {
    "severe_drought": 0.2,
    "moderate_drought": 0.4,
    "normal": 0.6,
    "wet": 0.8,
    "very_wet": 1.0
}

ENV_STRESS_SCALE = 0.5
MAX_PAIRS_PER_EPOCH = 25000
POINCARE_EPOCHS = 1000

USE_LOG_HYDRAULIC_METRIC = True
HYDRAULIC_LOG_EPS = 1e-12
DISTANCE_NORMALIZATION_PERCENTILE = 95.0

MRE_DISTANCE_FLOOR_PERCENTILE = 1.0

AMF_CONTACT_FRACTION = 0.10
MIN_AMF_CONTACTS = 10
MAX_AMF_CONTACTS = 50

PRESERVE_AMF_NODES = True
NORMALIZE_CONTACT_COORDINATES = True

AMF_NAME = "Glomus intraradices"
AMF_MODERN_NAME = "Rhizophagus irregularis"

AMF_NODES_CSV = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "AMF_network_nodes.csv"
)

AMF_EDGES_CSV = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "AMF_network_edges.csv"
)

OUTPUT_CSV_PATH = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "FINAL_EXP_RESULTS.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# RSML ROOT PARSING
# ============================================================

def validate_and_parse_rsml_functions(
    root_elem,
    ns=None
):
    functions_dict = {}

    def find_tag(elem, tag_name):
        return (
            elem.find(
                f"{{{ns}}}{tag_name}"
            )
            if ns
            else elem.find(tag_name)
        )

    def findall_tags(elem, tag_name):
        return (
            elem.findall(
                f"{{{ns}}}{tag_name}"
            )
            if ns
            else elem.findall(tag_name)
        )

    funcs_node = find_tag(
        root_elem,
        "functions"
    )

    if funcs_node is None:
        return functions_dict

    for func in findall_tags(
        funcs_node,
        "function"
    ):
        func_name = func.get(
            "name",
            ""
        ).lower()

        samples = []

        for sample in findall_tags(
            func,
            "sample"
        ):
            value = sample.get("value")

            if value is None:
                continue

            try:
                samples.append(
                    float(value)
                )
            except ValueError:
                continue

        functions_dict[
            func_name
        ] = samples

    return functions_dict


def parse_rsml_root(
    root_elem,
    parent_id=None,
    level=1,
    ns=None,
    plant_id="plant_0"
):
    def find_tag(elem, tag_name):
        return (
            elem.find(
                f"{{{ns}}}{tag_name}"
            )
            if ns
            else elem.find(tag_name)
        )

    def findall_tags(elem, tag_name):
        return (
            elem.findall(
                f"{{{ns}}}{tag_name}"
            )
            if ns
            else elem.findall(tag_name)
        )

    original_root_id = root_elem.get(
        "id",
        "unknown"
    )

    root_label = root_elem.get(
        "label",
        ""
    )

    root_id = (
        f"{plant_id}__{original_root_id}"
    )

    geometry = find_tag(
        root_elem,
        "geometry"
    )

    points = []

    if geometry is not None:

        polyline = find_tag(
            geometry,
            "polyline"
        )

        if polyline is not None:

            for point in findall_tags(
                polyline,
                "point"
            ):
                try:
                    points.append(
                        (
                            float(
                                point.get(
                                    "x",
                                    0
                                )
                            ),
                            float(
                                point.get(
                                    "y",
                                    0
                                )
                            ),
                            float(
                                point.get(
                                    "z",
                                    0
                                )
                            )
                        )
                    )

                except (
                    TypeError,
                    ValueError
                ):
                    continue

    funcs = (
        validate_and_parse_rsml_functions(
            root_elem,
            ns
        )
    )

    radius_values = funcs.get(
        "radius",
        []
    )

    if not radius_values:

        radius_values = [
            d / 2.0
            for d in funcs.get(
                "diameter",
                []
            )
        ]

    structured_points = []

    for index, position in enumerate(
        points
    ):

        if radius_values:

            radius = (
                radius_values[index]
                if index < len(
                    radius_values
                )
                else radius_values[-1]
            )

        else:

            radius = max(
                1.5 / level,
                0.1
            )

        structured_points.append({
            "index": index,
            "pos": position,
            "radius": max(
                float(radius),
                0.05
            )
        })

    roots_data = [{
        "id": root_id,
        "original_id": original_root_id,
        "label": root_label,
        "parent_id": parent_id,
        "level": level,
        "plant_id": plant_id,
        "points": structured_points
    }]

    for child in findall_tags(
        root_elem,
        "root"
    ):

        roots_data.extend(
            parse_rsml_root(
                child,
                parent_id=root_id,
                level=level + 1,
                ns=ns,
                plant_id=plant_id
            )
        )

    return roots_data


def parse_rsml_file(fpath):

    try:

        tree = ET.parse(
            fpath
        )

        xml_root = tree.getroot()

    except Exception as error:

        print(
            f"ERROR: "
            f"{os.path.basename(fpath)}: "
            f"{error}"
        )

        return []

    namespace = (
        xml_root.tag.split("}")[0].strip("{")
        if xml_root.tag.startswith("{")
        else None
    )

    if namespace:

        plants = xml_root.findall(
            f"{{{namespace}}}"
            f"scene/{{{namespace}}}plant"
        )

    else:

        plants = xml_root.findall(
            "scene/plant"
        )

    if not plants:
        plants = [
            xml_root
        ]

    all_roots = []

    for plant_index, plant in enumerate(
        plants
    ):

        plant_id = (
            f"plant_{plant_index}"
        )

        root_elements = (
            plant.findall(
                f"{{{namespace}}}root"
            )
            if namespace
            else plant.findall("root")
        )

        for root_element in root_elements:

            all_roots.extend(
                parse_rsml_root(
                    root_element,
                    ns=namespace,
                    plant_id=plant_id
                )
            )

    return all_roots


# ============================================================
# ENVIRONMENT + ROOT HYDRAULICS
# ============================================================

def environmental_architecture_scaling(
    roots_data,
    theta
):
    length_scale = (
        0.65
        + 0.35 * theta
    )

    radius_scale = (
        0.80
        + 0.20 * theta
    )

    transformed_map = {}
    transformed_roots = []

    for root in roots_data:

        original_points = root[
            "points"
        ]

        if not original_points:
            continue

        root_id = root["id"]
        parent_id = root["parent_id"]

        new_points = []

        if (
            parent_id is not None
            and parent_id in transformed_map
        ):

            parent_points = (
                transformed_map[parent_id]
            )

            original_start = np.array(
                original_points[0]["pos"],
                dtype=float
            )

            best_index = min(
                range(
                    len(parent_points)
                ),
                key=lambda i:
                    np.linalg.norm(
                        np.array(
                            parent_points[i][
                                "pos"
                            ],
                            dtype=float
                        )
                        - original_start
                    )
            )

            start_position = np.array(
                parent_points[
                    best_index
                ]["pos"],
                dtype=float
            )

        else:

            start_position = np.array(
                original_points[0]["pos"],
                dtype=float
            )

        new_points.append({
            "index":
                original_points[0]["index"],

            "pos":
                tuple(start_position),

            "radius":
                max(
                    original_points[0]["radius"]
                    * radius_scale,
                    0.05
                )
        })

        for i in range(
            1,
            len(original_points)
        ):

            previous_original = np.array(
                original_points[
                    i - 1
                ]["pos"],
                dtype=float
            )

            current_original = np.array(
                original_points[
                    i
                ]["pos"],
                dtype=float
            )

            direction = (
                current_original
                - previous_original
            )

            new_position = (
                np.array(
                    new_points[-1]["pos"],
                    dtype=float
                )
                + direction
                * length_scale
            )

            new_points.append({
                "index":
                    original_points[i]["index"],

                "pos":
                    tuple(new_position),

                "radius":
                    max(
                        original_points[i][
                            "radius"
                        ]
                        * radius_scale,
                        0.05
                    )
            })

        transformed_map[
            root_id
        ] = new_points

        transformed_roots.append({
            "id":
                root_id,

            "original_id":
                root.get(
                    "original_id",
                    root_id
                ),

            "label":
                root["label"],

            "parent_id":
                parent_id,

            "level":
                root["level"],

            "plant_id":
                root["plant_id"],

            "points":
                new_points
        })

    return transformed_roots


def environmental_factor(
    theta,
    radius,
    level
):
    vulnerability = (
        level
        / (radius + 0.1)
    )

    exponent = np.clip(
        (
            1.0 - theta
        )
        * vulnerability
        * ENV_STRESS_SCALE,
        -50.0,
        50.0
    )

    return np.exp(
        exponent
    )


def hydraulic_resistance(
    L,
    r,
    theta,
    level
):
    L = max(
        float(L),
        1e-8
    )

    r = max(
        float(r),
        0.05
    )

    return (
        (L / (r ** 4))
        * environmental_factor(
            theta,
            r,
            level
        )
    )


def build_bio_hydraulic_graph(
    roots_data,
    theta=1.0
):
    G = nx.Graph()

    root_node_map = {}

    node_counter = 0

    for root in roots_data:

        root_id = root["id"]

        points = root["points"]

        root_node_map[
            root_id
        ] = []

        for i, point in enumerate(
            points
        ):

            node_id = node_counter

            node_counter += 1

            G.add_node(
                node_id,

                root_id=root_id,

                original_root_id=root.get(
                    "original_id",
                    root_id
                ),

                plant_id=root[
                    "plant_id"
                ],

                level=root[
                    "level"
                ],

                pos=point["pos"],

                radius=max(
                    point["radius"],
                    0.05
                ),

                is_origin=(
                    root["level"] == 1
                    and i == 0
                ),

                network_type="root"
            )

            root_node_map[
                root_id
            ].append(
                node_id
            )

            if i == 0:
                continue

            previous_node = (
                root_node_map[
                    root_id
                ][i - 1]
            )

            position_a = np.array(
                G.nodes[
                    previous_node
                ]["pos"],
                dtype=float
            )

            position_b = np.array(
                point["pos"],
                dtype=float
            )

            length = float(
                np.linalg.norm(
                    position_a
                    - position_b
                )
            )

            radius = max(
                (
                    G.nodes[
                        previous_node
                    ]["radius"]
                    + point["radius"]
                ) / 2.0,
                0.05
            )

            resistance = (
                hydraulic_resistance(
                    length,
                    radius,
                    theta,
                    root["level"]
                )
            )

            G.add_edge(
                previous_node,
                node_id,

                hydraulic_resistance=
                    resistance,

                length=length,

                radius=radius,

                junction=False,

                edge_type="root"
            )

    for root in roots_data:

        parent_id = root[
            "parent_id"
        ]

        root_id = root["id"]

        if (
            parent_id not in root_node_map
            or not root_node_map[
                root_id
            ]
        ):
            continue

        child_first = (
            root_node_map[
                root_id
            ][0]
        )

        child_position = np.array(
            G.nodes[
                child_first
            ]["pos"],
            dtype=float
        )

        best_parent = None

        minimum_distance = float(
            "inf"
        )

        for parent_node in (
            root_node_map[
                parent_id
            ]
        ):

            parent_position = np.array(
                G.nodes[
                    parent_node
                ]["pos"],
                dtype=float
            )

            distance = float(
                np.linalg.norm(
                    child_position
                    - parent_position
                )
            )

            if distance < minimum_distance:

                minimum_distance = (
                    distance
                )

                best_parent = (
                    parent_node
                )

        if best_parent is None:
            continue

        radius = max(
            (
                G.nodes[
                    best_parent
                ]["radius"]
                + G.nodes[
                    child_first
                ]["radius"]
            ) / 2.0,
            0.05
        )

        resistance = (
            hydraulic_resistance(
                minimum_distance,
                radius,
                theta,
                root["level"]
            )
        )

        G.add_edge(
            best_parent,
            child_first,

            hydraulic_resistance=
                resistance,

            length=minimum_distance,

            radius=radius,

            junction=True,

            edge_type="root"
        )

    for _, _, data in G.edges(
        data=True
    ):

        resistance = max(
            float(
                data[
                    "hydraulic_resistance"
                ]
            ),
            HYDRAULIC_LOG_EPS
        )

        weight = (
            np.log1p(
                resistance
            )
            if USE_LOG_HYDRAULIC_METRIC
            else resistance
        )

        data["weight"] = max(
            float(weight),
            1e-8
        )

    return G


# ============================================================
# AMF INPUT
# ============================================================

def load_amf_inputs(
    nodes_csv,
    edges_csv
):
    for path in [
        nodes_csv,
        edges_csv
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"AMF input file not found: "
                f"{path}"
            )

    nodes = pd.read_csv(
        nodes_csv
    )

    edges = pd.read_csv(
        edges_csv
    )

    required_node_cols = [
        "amf_node_id",
        "x",
        "y",
        "z"
    ]

    required_edge_cols = [
        "source",
        "target",
        "length_mm"
    ]

    missing_nodes = [
        c
        for c in required_node_cols
        if c not in nodes.columns
    ]

    missing_edges = [
        c
        for c in required_edge_cols
        if c not in edges.columns
    ]

    if missing_nodes:

        raise ValueError(
            "Missing AMF node columns: "
            + ", ".join(
                missing_nodes
            )
        )

    if missing_edges:

        raise ValueError(
            "Missing AMF edge columns: "
            + ", ".join(
                missing_edges
            )
        )

    nodes = nodes.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    edges = edges.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    return nodes, edges


def _safe_float(
    value,
    default
):
    try:

        value = float(value)

        return (
            value
            if np.isfinite(value)
            else float(default)
        )

    except Exception:

        return float(default)


def build_amf_graph(
    nodes,
    edges,
    theta
):
    G = nx.Graph()

    if nodes.empty:
        return G

    base_colonization = 0.50

    if (
        "colonization_fraction"
        in nodes.columns
    ):

        s = pd.to_numeric(
            nodes[
                "colonization_fraction"
            ],
            errors="coerce"
        )

        if s.notna().any():

            base_colonization = float(
                s.dropna().median()
            )

    base_hld = 1.0

    if (
        "hyphal_length_density_m_per_g"
        in nodes.columns
    ):

        s = pd.to_numeric(
            nodes[
                "hyphal_length_density_m_per_g"
            ],
            errors="coerce"
        )

        if s.notna().any():

            base_hld = float(
                s.dropna().median()
            )

    environmental_activity = (
        0.50
        + 0.50 * float(theta)
    )

    for _, row in nodes.iterrows():

        node = int(
            row["amf_node_id"]
        )

        radius = _safe_float(
            row.get(
                "hyphal_radius_um",
                0.5
            ),
            0.5
        )

        branch_order = int(
            _safe_float(
                row.get(
                    "branch_order",
                    1
                ),
                1
            )
        )

        colonization = _safe_float(
            row.get(
                "colonization_fraction",
                base_colonization
            ),
            base_colonization
        )

        colonization = float(
            np.clip(
                colonization
                * environmental_activity,
                0.0,
                1.0
            )
        )

        hld = max(
            _safe_float(
                row.get(
                    "hyphal_length_density_m_per_g",
                    base_hld
                ),
                base_hld
            ),
            0.0
        )

        p_capacity = max(
            _safe_float(
                row.get(
                    "P_transport_capacity_relative",
                    1.0
                ),
                1.0
            ),
            0.0
        )

        n_capacity = max(
            _safe_float(
                row.get(
                    "N_transport_capacity_relative",
                    1.0
                ),
                1.0
            ),
            0.0
        )

        G.add_node(
            node,

            network_type="amf",

            fungus=row.get(
                "fungus",
                AMF_NAME
            ),

            fungus_modern=AMF_MODERN_NAME,

            host=row.get(
                "host",
                "Zea mays"
            ),

            network_id=row.get(
                "network_id",
                "AMF_1"
            ),

            pos=(
                _safe_float(
                    row.get("x", 0.0),
                    0.0
                ),

                _safe_float(
                    row.get("y", 0.0),
                    0.0
                ),

                _safe_float(
                    row.get("z", 0.0),
                    0.0
                )
            ),

            radius=max(
                radius,
                0.1
            ),

            branch_order=
                branch_order,

            colonization_fraction=
                colonization,

            hyphal_length_density=
                hld,

            P_capacity=
                p_capacity,

            N_capacity=
                n_capacity
        )

    for _, row in edges.iterrows():

        u = int(
            _safe_float(
                row.get("source"),
                -1
            )
        )

        v = int(
            _safe_float(
                row.get("target"),
                -1
            )
        )

        if (
            u not in G
            or v not in G
            or u == v
        ):
            continue

        length = max(
            _safe_float(
                row.get(
                    "length_mm",
                    1.0
                ),
                1.0
            ),
            1e-6
        )

        default_radius = (
            G.nodes[u]["radius"]
            + G.nodes[v]["radius"]
        ) / 2.0

        radius_um = max(
            _safe_float(
                row.get(
                    "mean_hyphal_radius_um",
                    default_radius
                ),
                default_radius
            ),
            0.1
        )

        branch_order_target = int(
            _safe_float(
                row.get(
                    "branch_order_target",
                    G.nodes[v].get(
                        "branch_order",
                        1
                    )
                ),
                1
            )
        )

        resistance = (
            length
            / (radius_um ** 2)
            / max(
                environmental_activity,
                1e-8
            )
        )

        G.add_edge(
            u,
            v,

            length=length,

            radius=radius_um,

            branch_order_target=
                branch_order_target,

            hydraulic_resistance=
                max(
                    resistance,
                    1e-12
                ),

            weight=np.log1p(
                max(
                    resistance,
                    1e-12
                )
            ),

            edge_type="amf"
        )

    return G


def _normalized_contact_positions(G):

    positions = {
        n: np.asarray(
            a.get(
                "pos",
                (
                    0.0,
                    0.0,
                    0.0
                )
            ),
            dtype=float
        )
        for n, a in G.nodes(
            data=True
        )
    }

    if not positions:
        return positions

    matrix = np.vstack(
        list(
            positions.values()
        )
    )

    center = np.mean(
        matrix,
        axis=0
    )

    centered = (
        matrix - center
    )

    scale = float(
        np.max(
            np.linalg.norm(
                centered,
                axis=1
            )
        )
    )

    if (
        not np.isfinite(scale)
        or scale <= 1e-12
    ):
        scale = 1.0

    return {
        n: (
            p - center
        ) / scale
        for n, p in positions.items()
    }


def attach_amf_to_roots(
    root_G,
    amf_G,
    theta
):
    G = nx.Graph()

    G.add_nodes_from(
        (
            n,
            dict(attrs)
        )
        for n, attrs
        in root_G.nodes(
            data=True
        )
    )

    G.add_edges_from(
        (
            u,
            v,
            dict(attrs)
        )
        for u, v, attrs
        in root_G.edges(
            data=True
        )
    )

    offset = (
        max(G.nodes) + 1
        if G.nodes
        else 0
    )

    amf_mapping = {
        n: n + offset
        for n in amf_G.nodes
    }

    for n, attrs in amf_G.nodes(
        data=True
    ):

        attrs2 = dict(attrs)

        attrs2[
            "amf_original_node"
        ] = n

        G.add_node(
            amf_mapping[n],
            **attrs2
        )

    for u, v, attrs in amf_G.edges(
        data=True
    ):

        G.add_edge(
            amf_mapping[u],
            amf_mapping[v],
            **dict(attrs)
        )

    root_nodes = [
        n
        for n, a in G.nodes(
            data=True
        )
        if a.get(
            "network_type"
        ) == "root"
    ]

    fungal_nodes = [
        n
        for n, a in G.nodes(
            data=True
        )
        if a.get(
            "network_type"
        ) == "amf"
    ]

    if (
        not root_nodes
        or not fungal_nodes
    ):
        return G, 0

    if NORMALIZE_CONTACT_COORDINATES:

        root_pos = (
            _normalized_contact_positions(
                G.subgraph(
                    root_nodes
                ).copy()
            )
        )

        amf_pos = (
            _normalized_contact_positions(
                G.subgraph(
                    fungal_nodes
                ).copy()
            )
        )

    else:

        root_pos = {
            n: np.asarray(
                G.nodes[n]["pos"],
                dtype=float
            )
            for n in root_nodes
        }

        amf_pos = {
            n: np.asarray(
                G.nodes[n]["pos"],
                dtype=float
            )
            for n in fungal_nodes
        }

    root_matrix = np.vstack([
        root_pos[n]
        for n in root_nodes
    ])

    components = list(
        nx.connected_components(
            G.subgraph(
                fungal_nodes
            )
        )
    )

    component_count = len(
        components
    )

    max_contacts = min(
        len(fungal_nodes),
        len(root_nodes),
        max(
            component_count,
            MIN_AMF_CONTACTS,
            min(
                MAX_AMF_CONTACTS,
                int(
                    AMF_CONTACT_FRACTION
                    * len(
                        fungal_nodes
                    )
                )
            )
        )
    )

    selected = []
    selected_set = set()

    for component in components:

        component_nodes = list(
            component
        )

        best_node = min(
            component_nodes,
            key=lambda fn:
                float(
                    np.min(
                        np.linalg.norm(
                            root_matrix
                            - amf_pos[fn],
                            axis=1
                        )
                    )
                )
        )

        if best_node not in selected_set:

            selected.append(
                best_node
            )

            selected_set.add(
                best_node
            )

    remaining = [
        fn
        for fn in fungal_nodes
        if fn not in selected_set
    ]

    remaining.sort(
        key=lambda fn:
            float(
                np.min(
                    np.linalg.norm(
                        root_matrix
                        - amf_pos[fn],
                        axis=1
                    )
                )
            )
    )

    for fn in remaining:

        if len(selected) >= max_contacts:
            break

        selected.append(fn)

        selected_set.add(fn)

    contacts = 0

    for fn in selected:

        fp = amf_pos[fn]

        distances = np.linalg.norm(
            root_matrix - fp,
            axis=1
        )

        root_index = int(
            np.argmin(
                distances
            )
        )

        rn = root_nodes[
            root_index
        ]

        d = max(
            float(
                distances[
                    root_index
                ]
            ),
            1e-6
        )

        colonization = float(
            np.clip(
                G.nodes[fn].get(
                    "colonization_fraction",
                    0.5
                ),
                0.0,
                1.0
            )
        )

        coupling_resistance = (
            d
            / max(
                colonization
                * (
                    0.50
                    + 0.50 * float(theta)
                ),
                1e-6
            )
        )

        if G.has_edge(
            rn,
            fn
        ):
            continue

        G.add_edge(
            rn,
            fn,

            length=d,

            radius=1.0,

            hydraulic_resistance=max(
                coupling_resistance,
                1e-12
            ),

            weight=np.log1p(
                max(
                    coupling_resistance,
                    1e-12
                )
            ),

            edge_type=
                "symbiotic_interface",

            colonization=
                colonization
        )

        contacts += 1

    return G, contacts


# ============================================================
# COARSENING + DISTANCES
# ============================================================

def bio_hydraulic_coarsening(
    G,
    target_max=400
):
    if G.number_of_nodes() <= target_max:
        return G.copy()

    G_c = G.copy()

    protected = set()

    if PRESERVE_AMF_NODES:

        protected.update(
            n
            for n, a in G_c.nodes(
                data=True
            )
            if a.get(
                "network_type"
            ) == "amf"
        )

    for u, v, a in G_c.edges(
        data=True
    ):

        if a.get(
            "edge_type"
        ) == "symbiotic_interface":

            protected.add(u)
            protected.add(v)

    protected.update(
        n
        for n, a in G_c.nodes(
            data=True
        )
        if a.get(
            "is_origin",
            False
        )
    )

    def removable_nodes():

        return [
            n
            for n in G_c.nodes()
            if (
                n not in protected
                and G_c.nodes[n].get(
                    "network_type"
                ) == "root"
            )
        ]

    while (
        G_c.number_of_nodes()
        > target_max
    ):

        candidates = [
            n
            for n in removable_nodes()
            if G_c.degree(n) == 2
        ]

        if candidates:

            node = min(
                candidates,
                key=lambda n: (
                    G_c.nodes[n].get(
                        "level",
                        999999
                    ),
                    G_c.nodes[n].get(
                        "radius",
                        999999
                    )
                )
            )

            u, v = list(
                G_c.neighbors(node)
            )

            e1 = dict(
                G_c[u][node]
            )

            e2 = dict(
                G_c[node][v]
            )

            w1 = float(
                e1.get(
                    "weight",
                    1.0
                )
            )

            w2 = float(
                e2.get(
                    "weight",
                    1.0
                )
            )

            l1 = float(
                e1.get(
                    "length",
                    0.0
                )
            )

            l2 = float(
                e2.get(
                    "length",
                    0.0
                )
            )

            r1 = _safe_float(
                e1.get(
                    "hydraulic_resistance",
                    np.nan
                ),
                np.nan
            )

            r2 = _safe_float(
                e2.get(
                    "hydraulic_resistance",
                    np.nan
                ),
                np.nan
            )

            if not np.isfinite(r1):

                r1 = max(
                    w1,
                    1e-12
                )

            if not np.isfinite(r2):

                r2 = max(
                    w2,
                    1e-12
                )

            combined = {
                "weight":
                    max(
                        w1 + w2,
                        1e-12
                    ),

                "length":
                    max(
                        l1 + l2,
                        0.0
                    ),

                "hydraulic_resistance":
                    max(
                        r1 + r2,
                        1e-12
                    ),

                "radius":
                    max(
                        (
                            float(
                                e1.get(
                                    "radius",
                                    0.1
                                )
                            )
                            * max(
                                l1,
                                0.0
                            )
                            +
                            float(
                                e2.get(
                                    "radius",
                                    0.1
                                )
                            )
                            * max(
                                l2,
                                0.0
                            )
                        )
                        / max(
                            l1 + l2,
                            1e-12
                        ),
                        0.05
                    ),

                "edge_type":
                    "root",

                "junction":
                    bool(
                        e1.get(
                            "junction",
                            False
                        )
                        or e2.get(
                            "junction",
                            False
                        )
                    )
            }

            G_c.remove_node(
                node
            )

            if G_c.has_edge(
                u,
                v
            ):

                if (
                    G_c[u][v].get(
                        "edge_type"
                    )
                    !=
                    "symbiotic_interface"
                ):

                    if (
                        combined["weight"]
                        <
                        G_c[u][v].get(
                            "weight",
                            np.inf
                        )
                    ):

                        G_c[u][v].update(
                            combined
                        )

            else:

                G_c.add_edge(
                    u,
                    v,
                    **combined
                )

            continue

        candidates = (
            removable_nodes()
        )

        if not candidates:
            break

        node = min(
            candidates,
            key=lambda n: (
                G_c.degree(n),
                G_c.nodes[n].get(
                    "level",
                    999999
                ),
                G_c.nodes[n].get(
                    "radius",
                    999999
                )
            )
        )

        neighbors = list(
            G_c.neighbors(node)
        )

        if len(neighbors) <= 1:

            G_c.remove_node(
                node
            )

            continue

        for i in range(
            len(neighbors)
        ):

            for j in range(
                i + 1,
                len(neighbors)
            ):

                u = neighbors[i]
                v = neighbors[j]

                e1 = G_c[u][node]
                e2 = G_c[node][v]

                w = (
                    float(
                        e1.get(
                            "weight",
                            1.0
                        )
                    )
                    +
                    float(
                        e2.get(
                            "weight",
                            1.0
                        )
                    )
                )

                l = (
                    float(
                        e1.get(
                            "length",
                            0.0
                        )
                    )
                    +
                    float(
                        e2.get(
                            "length",
                            0.0
                        )
                    )
                )

                r1 = _safe_float(
                    e1.get(
                        "hydraulic_resistance",
                        np.nan
                    ),
                    np.nan
                )

                r2 = _safe_float(
                    e2.get(
                        "hydraulic_resistance",
                        np.nan
                    ),
                    np.nan
                )

                if not np.isfinite(r1):

                    r1 = max(
                        float(
                            e1.get(
                                "weight",
                                1.0
                            )
                        ),
                        1e-12
                    )

                if not np.isfinite(r2):

                    r2 = max(
                        float(
                            e2.get(
                                "weight",
                                1.0
                            )
                        ),
                        1e-12
                    )

                combined = {
                    "weight":
                        max(
                            w,
                            1e-12
                        ),

                    "length":
                        max(
                            l,
                            0.0
                        ),

                    "hydraulic_resistance":
                        max(
                            r1 + r2,
                            1e-12
                        ),

                    "radius":
                        max(
                            (
                                float(
                                    e1.get(
                                        "radius",
                                        0.1
                                    )
                                )
                                +
                                float(
                                    e2.get(
                                        "radius",
                                        0.1
                                    )
                                )
                            ) / 2.0,
                            0.05
                        ),

                    "edge_type":
                        "root",

                    "junction":
                        bool(
                            e1.get(
                                "junction",
                                False
                            )
                            or
                            e2.get(
                                "junction",
                                False
                            )
                        )
                }

                if G_c.has_edge(
                    u,
                    v
                ):

                    if (
                        G_c[u][v].get(
                            "edge_type"
                        )
                        !=
                        "symbiotic_interface"
                    ):

                        if (
                            combined["weight"]
                            <
                            G_c[u][v].get(
                                "weight",
                                np.inf
                            )
                        ):

                            G_c[u][v].update(
                                combined
                            )

                else:

                    G_c.add_edge(
                        u,
                        v,
                        **combined
                    )

        G_c.remove_node(
            node
        )

    return G_c


def normalize_graph_distances(
    D_graph
):
    upper = D_graph[
        np.triu_indices_from(
            D_graph,
            k=1
        )
    ]

    positive = upper[
        upper > 0
    ]

    if len(positive) == 0:
        return None

    normalization_value = np.percentile(
        positive,
        DISTANCE_NORMALIZATION_PERCENTILE
    )

    if (
        not np.isfinite(
            normalization_value
        )
        or normalization_value <= 0
    ):

        normalization_value = np.max(
            positive
        )

    return np.maximum(
        D_graph
        / (
            normalization_value
            + 1e-12
        ),
        0.0
    )


def prepare_graph(
    roots_data,
    amf_nodes,
    amf_edges,
    theta
):
    transformed_roots = (
        environmental_architecture_scaling(
            roots_data,
            theta
        )
    )

    root_G = (
        build_bio_hydraulic_graph(
            transformed_roots,
            theta
        )
    )

    if root_G.number_of_nodes() < 5:

        return None, None, 0

    amf_G = (
        build_amf_graph(
            amf_nodes,
            amf_edges,
            theta
        )
    )

    coupled_G, contacts = (
        attach_amf_to_roots(
            root_G,
            amf_G,
            theta
        )
    )

    G = (
        bio_hydraulic_coarsening(
            coupled_G,
            TARGET_MAX_NODES
        )
    )

    if G.number_of_nodes() < 5:

        return None, None, contacts

    if not nx.is_connected(G):

        largest_component = max(
            nx.connected_components(G),
            key=len
        )

        G = G.subgraph(
            largest_component
        ).copy()

    if not nx.is_connected(G):

        return None, None, contacts

    G = (
        nx.convert_node_labels_to_integers(
            G,
            label_attribute=
                "original_node_id"
        )
    )

    adjacency = (
        nx.to_scipy_sparse_array(
            G,
            weight="weight",
            dtype=np.float64
        )
    )

    D_graph = dijkstra(
        csgraph=adjacency,
        directed=False,
        return_predecessors=False
    ).astype(
        np.float64
    )

    if not np.all(
        np.isfinite(D_graph)
    ):

        return None, None, contacts

    D_graph = (
        normalize_graph_distances(
            D_graph
        )
    )

    if D_graph is None:

        return None, None, contacts

    return (
        G,
        D_graph,
        contacts
    )


# ============================================================
# EMBEDDINGS
# ============================================================

def run_poincare(
    D_graph,
    dim=2,
    seed=42,
    max_epochs=1000
):
    set_seed(seed)

    N = D_graph.shape[0]

    D_tensor = torch.tensor(
        D_graph,
        dtype=torch.float32,
        device=device
    )

    upper_u, upper_v = np.triu_indices(
        N,
        k=1
    )

    num_pairs = len(
        upper_u
    )

    if num_pairs <= MAX_PAIRS_PER_EPOCH:

        u_full = torch.tensor(
            upper_u,
            dtype=torch.long,
            device=device
        )

        v_full = torch.tensor(
            upper_v,
            dtype=torch.long,
            device=device
        )

    else:

        u_full = None
        v_full = None

    mds = MDS(
        n_components=dim,
        dissimilarity="precomputed",
        random_state=seed,
        max_iter=300,
        n_init=1
    )

    init_coordinates = (
        mds.fit_transform(
            D_graph
        )
    )

    max_norm = np.max(
        np.linalg.norm(
            init_coordinates,
            axis=1
        )
    )

    init_coordinates = (
        init_coordinates
        / (max_norm + 1e-8)
    ) * 0.8

    embeddings = torch.nn.Parameter(
        torch.tensor(
            init_coordinates,
            dtype=torch.float32,
            device=device
        )
    )

    curvature_raw = torch.nn.Parameter(
        torch.tensor(
            [0.0],
            dtype=torch.float32,
            device=device
        )
    )

    lr_embedding = (
        0.005
        / np.sqrt(
            max(
                1.0,
                dim / 2.0
            )
        )
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    embeddings,
                "lr":
                    lr_embedding
            },
            {
                "params":
                    curvature_raw,
                "lr":
                    0.001
            }
        ]
    )

    for _ in range(
        max_epochs
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        curvature = (
            F.softplus(
                curvature_raw
            )
            + 1e-4
        )

        sqrt_curvature = (
            torch.sqrt(
                curvature
            )
        )

        if num_pairs > MAX_PAIRS_PER_EPOCH:

            sampled = np.random.choice(
                num_pairs,
                MAX_PAIRS_PER_EPOCH,
                replace=False
            )

            u = torch.tensor(
                upper_u[sampled],
                dtype=torch.long,
                device=device
            )

            v = torch.tensor(
                upper_v[sampled],
                dtype=torch.long,
                device=device
            )

        else:

            u = u_full
            v = v_full

        x = embeddings[u]
        y = embeddings[v]

        diff_sq = torch.sum(
            (x - y) ** 2,
            dim=-1
        )

        x_sq = torch.sum(
            x ** 2,
            dim=-1
        )

        y_sq = torch.sum(
            y ** 2,
            dim=-1
        )

        denominator = torch.clamp(
            (
                1.0
                - curvature * x_sq
            )
            *
            (
                1.0
                - curvature * y_sq
            ),
            min=1e-6
        )

        alpha = torch.clamp(
            1.0
            + (
                2.0
                * curvature
                * diff_sq
                / denominator
            ),
            min=1.0 + 1e-6
        )

        hyperbolic_distance = (
            2.0
            / sqrt_curvature
        ) * torch.arccosh(
            alpha
        )

        target = D_tensor[
            u,
            v
        ]

        scale = (
            torch.sum(
                hyperbolic_distance
                * target
            )
            / (
                torch.sum(
                    hyperbolic_distance ** 2
                )
                + 1e-8
            )
        )

        loss = F.smooth_l1_loss(
            scale.detach()
            * hyperbolic_distance,
            target
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [
                embeddings,
                curvature_raw
            ],
            max_norm=1.0
        )

        optimizer.step()

        with torch.no_grad():

            curvature = (
                F.softplus(
                    curvature_raw
                )
                + 1e-4
            )

            sqrt_curvature = (
                torch.sqrt(
                    curvature
                )
            )

            norms = embeddings.norm(
                dim=-1,
                keepdim=True
            )

            boundary = (
                0.999
                / sqrt_curvature
            )

            embeddings.data = torch.where(
                norms >= boundary,
                embeddings.data
                / (
                    norms
                    + 1e-12
                )
                * boundary,
                embeddings.data
            )

    embedding_np = (
        embeddings
        .detach()
        .cpu()
        .numpy()
    )

    final_curvature = (
        F.softplus(
            curvature_raw
        )
        + 1e-4
    ).item()

    x_sq = np.sum(
        embedding_np ** 2,
        axis=1,
        keepdims=True
    )

    difference_sq = np.sum(
        (
            embedding_np[:, None, :]
            - embedding_np[None, :, :]
        ) ** 2,
        axis=-1
    )

    denominator = (
        1.0
        - final_curvature * x_sq
    )

    denominator = np.clip(
        denominator @ denominator.T,
        1e-6,
        None
    )

    alpha = np.clip(
        1.0
        + (
            2.0
            * final_curvature
            * difference_sq
            / denominator
        ),
        1.0 + 1e-6,
        None
    )

    D_raw = (
        2.0
        / np.sqrt(
            final_curvature
        )
    ) * np.arccosh(
        alpha
    )

    scale = (
        np.sum(
            D_raw * D_graph
        )
        / (
            np.sum(
                D_raw ** 2
            )
            + 1e-12
        )
    )

    D_pred = (
        scale
        * D_raw
    )

    return (
        embedding_np,
        D_pred,
        final_curvature
    )


def run_mds(
    D_graph,
    dim=2,
    seed=42
):
    model = MDS(
        n_components=dim,
        dissimilarity="precomputed",
        random_state=seed,
        max_iter=500,
        n_init=4
    )

    embedding_np = (
        model.fit_transform(
            D_graph
        )
    )

    D_raw = np.linalg.norm(
        embedding_np[:, None, :]
        - embedding_np[None, :, :],
        axis=-1
    )

    scale = (
        np.sum(
            D_raw * D_graph
        )
        / (
            np.sum(
                D_raw ** 2
            )
            + 1e-12
        )
    )

    return (
        embedding_np,
        scale * D_raw
    )


def run_spectral(
    D_graph,
    dim=2,
    seed=42
):
    positive = D_graph[
        D_graph > 0
    ]

    median_distance = (
        np.median(
            positive
        )
        + 1e-8
        if len(positive)
        else 1.0
    )

    similarity = np.exp(
        -D_graph
        / median_distance
    )

    model = SpectralEmbedding(
        n_components=dim,
        affinity="precomputed",
        random_state=seed
    )

    embedding_np = (
        model.fit_transform(
            similarity
        )
    )

    D_raw = np.linalg.norm(
        embedding_np[:, None, :]
        - embedding_np[None, :, :],
        axis=-1
    )

    scale = (
        np.sum(
            D_raw * D_graph
        )
        / (
            np.sum(
                D_raw ** 2
            )
            + 1e-12
        )
    )

    return (
        embedding_np,
        scale * D_raw
    )


# ============================================================
# METRICS
# ============================================================

def hyperbolic_distances_from_origin(
    embedding,
    origin,
    curvature
):
    x_sq = np.sum(
        embedding ** 2,
        axis=1
    )

    origin_sq = np.sum(
        origin ** 2
    )

    difference_sq = np.sum(
        (
            embedding - origin
        ) ** 2,
        axis=1
    )

    denominator = np.clip(
        (
            1.0
            - curvature * x_sq
        )
        *
        (
            1.0
            - curvature * origin_sq
        ),
        1e-6,
        None
    )

    alpha = np.clip(
        1.0
        + (
            2.0
            * curvature
            * difference_sq
            / denominator
        ),
        1.0 + 1e-6,
        None
    )

    return (
        2.0
        / np.sqrt(
            curvature
        )
    ) * np.arccosh(
        alpha
    )


def compute_hierarchy_metric(
    G,
    embedding,
    is_poincare=False,
    poincare_c=None
):
    origins = [
        n
        for n, a in G.nodes(
            data=True
        )
        if a.get(
            "is_origin",
            False
        )
    ]

    values = []

    for origin in origins:

        plant_id = G.nodes[
            origin
        ].get(
            "plant_id"
        )

        plant_nodes = [
            n
            for n, a in G.nodes(
                data=True
            )
            if (
                a.get(
                    "plant_id"
                )
                == plant_id
                and a.get(
                    "network_type"
                ) == "root"
            )
        ]

        if len(
            plant_nodes
        ) < 3:
            continue

        component = (
            nx.node_connected_component(
                G,
                origin
            )
        )

        indices = sorted(
            set(
                plant_nodes
            ).intersection(
                component
            )
        )

        if len(
            indices
        ) < 3:
            continue

        levels = np.array(
            [
                G.nodes[n]["level"]
                for n in indices
            ],
            dtype=float
        )

        if len(
            np.unique(levels)
        ) < 2:
            continue

        local_embedding = (
            embedding[indices]
        )

        origin_embedding = (
            embedding[origin]
        )

        if (
            is_poincare
            and poincare_c is not None
        ):

            distances = (
                hyperbolic_distances_from_origin(
                    local_embedding,
                    origin_embedding,
                    poincare_c
                )
            )

        else:

            distances = np.linalg.norm(
                local_embedding
                - origin_embedding,
                axis=1
            )

        correlation, _ = (
            stats.spearmanr(
                levels,
                distances
            )
        )

        if np.isfinite(
            correlation
        ):

            values.append(
                float(correlation)
            )

    return (
        float(
            np.mean(values)
        )
        if values
        else 0.0
    )


def embedding_edge_distances(
    G,
    D_true,
    D_pred
):
    records = []

    for u, v, data in G.edges(
        data=True
    ):

        true_d = float(
            D_true[u, v]
        )

        pred_d = float(
            D_pred[u, v]
        )

        if (
            not np.isfinite(
                true_d
            )
            or not np.isfinite(
                pred_d
            )
        ):
            continue

        if true_d <= 0:
            continue

        records.append(
            (
                u,
                v,
                true_d,
                max(
                    pred_d,
                    1e-12
                )
            )
        )

    return records


def compute_embedding_dependent_biological_metrics(
    G,
    D_true,
    D_pred,
    theta
):
    edge_records = (
        embedding_edge_distances(
            G,
            D_true,
            D_pred
        )
    )

    if not edge_records:

        return {
            "Physical_Hydraulic":
                0.0,

            "Hydraulic":
                0.0,

            "Hydraulic_Preservation":
                0.0,

            "Embedding_Transport_Preservation":
                0.0,

            "Nutrient_Uptake":
                0.0,

            "Nutrient_Final":
                0.0,

            "Nutrient_Flux":
                0.0
        }

    physical_resistances = []

    for _, _, data in G.edges(
        data=True
    ):

        r = _safe_float(
            data.get(
                "hydraulic_resistance",
                np.nan
            ),
            np.nan
        )

        if (
            not np.isfinite(r)
            or r <= 0
        ):

            length = max(
                _safe_float(
                    data.get(
                        "length",
                        1.0
                    ),
                    1.0
                ),
                1e-8
            )

            radius = max(
                _safe_float(
                    data.get(
                        "radius",
                        1.0
                    ),
                    1.0
                ),
                0.05
            )

            r = (
                length
                / (radius ** 4)
            )

        if (
            np.isfinite(r)
            and r > 0
        ):

            physical_resistances.append(
                float(r)
            )

    physical_hydraulic = (
        float(
            np.mean(
                physical_resistances
            )
        )
        if physical_resistances
        else 0.0
    )

    adjusted_resistances = []
    preservation_scores = []

    for (
        u,
        v,
        true_d,
        pred_d
    ) in edge_records:

        distortion = float(
            np.clip(
                pred_d
                / max(
                    true_d,
                    1e-12
                ),
                0.10,
                10.0
            )
        )

        base_r = _safe_float(
            G[u][v].get(
                "hydraulic_resistance",
                np.nan
            ),
            np.nan
        )

        if (
            not np.isfinite(
                base_r
            )
            or base_r <= 0
        ):

            length = max(
                _safe_float(
                    G[u][v].get(
                        "length",
                        1.0
                    ),
                    1.0
                ),
                1e-8
            )

            radius = max(
                _safe_float(
                    G[u][v].get(
                        "radius",
                        1.0
                    ),
                    1.0
                ),
                0.05
            )

            base_r = (
                length
                / (radius ** 4)
            )

        adjusted_r = max(
            float(base_r)
            * distortion,
            1e-12
        )

        adjusted_resistances.append(
            adjusted_r
        )

        preservation = (
            1.0
            / (
                1.0
                + abs(
                    np.log(
                        distortion
                    )
                )
            )
        )

        preservation_scores.append(
            preservation
        )

    hydraulic_proxy = float(
        np.mean(
            adjusted_resistances
        )
    )

    transport_preservation = float(
        np.mean(
            preservation_scores
        )
    )

    hydraulic_preservation = float(
        np.clip(
            physical_hydraulic
            / max(
                hydraulic_proxy,
                1e-12
            ),
            0.0,
            1.0
        )
    )

    interfaces = [
        (
            u,
            v,
            a
        )
        for u, v, a
        in G.edges(
            data=True
        )
        if a.get(
            "edge_type"
        ) == "symbiotic_interface"
    ]

    amf_nodes = [
        (
            n,
            a
        )
        for n, a
        in G.nodes(
            data=True
        )
        if a.get(
            "network_type"
        ) == "amf"
    ]

    if (
        not interfaces
        or not amf_nodes
    ):

        return {
            "Physical_Hydraulic":
                physical_hydraulic,

            "Hydraulic":
                hydraulic_proxy,

            "Hydraulic_Preservation":
                hydraulic_preservation,

            "Embedding_Transport_Preservation":
                transport_preservation,

            "Nutrient_Uptake":
                0.0,

            "Nutrient_Final":
                0.0,

            "Nutrient_Flux":
                0.0
        }

    interface_scores = []

    for (
        u,
        v,
        attrs
    ) in interfaces:

        true_d = float(
            D_true[u, v]
        )

        pred_d = float(
            D_pred[u, v]
        )

        if (
            not np.isfinite(
                true_d
            )
            or not np.isfinite(
                pred_d
            )
            or true_d <= 0
        ):
            continue

        distortion = float(
            np.clip(
                pred_d
                / max(
                    true_d,
                    1e-12
                ),
                0.10,
                10.0
            )
        )

        preservation = (
            1.0
            / (
                1.0
                + abs(
                    np.log(
                        distortion
                    )
                )
            )
        )

        amf_node = (
            v
            if G.nodes[v].get(
                "network_type"
            ) == "amf"
            else u
        )

        amf_attr = G.nodes[
            amf_node
        ]

        colonization = float(
            np.clip(
                _safe_float(
                    amf_attr.get(
                        "colonization_fraction",
                        0.5
                    ),
                    0.5
                ),
                0.0,
                1.0
            )
        )

        p_capacity = max(
            _safe_float(
                amf_attr.get(
                    "P_capacity",
                    1.0
                ),
                1.0
            ),
            0.0
        )

        n_capacity = max(
            _safe_float(
                amf_attr.get(
                    "N_capacity",
                    1.0
                ),
                1.0
            ),
            0.0
        )

        interface_resistance = max(
            _safe_float(
                attrs.get(
                    "hydraulic_resistance",
                    1.0
                ),
                1.0
            ),
            1e-12
        )

        interface_conductance = (
            1.0
            / interface_resistance
        )

        nutrient_capacity = (
            0.70 * p_capacity
            + 0.30 * n_capacity
        )

        interface_scores.append(
            colonization
            * nutrient_capacity
            * interface_conductance
            * preservation
        )

    if not interface_scores:

        flux = 0.0

    else:

        raw_flux = float(
            np.sum(
                interface_scores
            )
        )

        flux = (
            raw_flux
            / max(
                len(interfaces),
                1
            )
            * (
                0.50
                + 0.50 * float(theta)
            )
            * transport_preservation
        )

    flux = float(
        max(
            flux,
            0.0
        )
    )

    time = np.arange(
        0.0,
        1.000001,
        0.1
    )

    uptake_rate = (
        3.0 * flux
    )

    uptake_curve = (
        1.0
        - np.exp(
            -uptake_rate * time
        )
    )

    final_uptake = float(
        np.clip(
            uptake_curve[-1],
            0.0,
            1.0
        )
    )

    try:

        uptake_auc = float(
            np.trapezoid(
                uptake_curve,
                time
            )
        )

    except AttributeError:

        uptake_auc = float(
            np.trapz(
                uptake_curve,
                time
            )
        )

    return {
        "Physical_Hydraulic":
            physical_hydraulic,

        "Hydraulic":
            hydraulic_proxy,

        "Hydraulic_Preservation":
            hydraulic_preservation,

        "Embedding_Transport_Preservation":
            transport_preservation,

        "Nutrient_Uptake":
            uptake_auc,

        "Nutrient_Final":
            final_uptake,

        "Nutrient_Flux":
            flux
    }


def compute_metrics(
    G,
    embedding,
    D_true,
    D_pred,
    theta,
    is_poincare=False,
    poincare_c=None
):
    indices = np.triu_indices_from(
        D_true,
        k=1
    )

    true_values = np.asarray(
        D_true[indices],
        dtype=float
    )

    predicted_values = np.asarray(
        D_pred[indices],
        dtype=float
    )

    valid = (
        np.isfinite(true_values)
        & np.isfinite(
            predicted_values
        )
        & (true_values > 0)
    )

    true_values = (
        true_values[valid]
    )

    predicted_values = (
        predicted_values[valid]
    )

    if len(
        true_values
    ) == 0:

        mre = np.nan
        raw_mre = np.nan
        stress = np.nan

    else:

        floor = float(
            np.percentile(
                true_values,
                MRE_DISTANCE_FLOOR_PERCENTILE
            )
        )

        floor = max(
            floor,
            1e-8
        )

        raw_mre = float(
            np.mean(
                np.abs(
                    predicted_values
                    - true_values
                )
                / np.maximum(
                    true_values,
                    1e-12
                )
            )
        )

        mre = float(
            np.mean(
                np.abs(
                    predicted_values
                    - true_values
                )
                / np.maximum(
                    true_values,
                    floor
                )
            )
        )

        stress = float(
            np.sqrt(
                np.sum(
                    (
                        predicted_values
                        - true_values
                    ) ** 2
                )
                / (
                    np.sum(
                        true_values ** 2
                    )
                    + 1e-12
                )
            )
        )

    hierarchy = (
        compute_hierarchy_metric(
            G,
            embedding,
            is_poincare=
                is_poincare,
            poincare_c=
                poincare_c
        )
    )

    biological = (
        compute_embedding_dependent_biological_metrics(
            G,
            D_true,
            D_pred,
            theta
        )
    )

    return {
        "MRE":
            mre,

        "Raw_MRE":
            raw_mre,

        "Stress":
            stress,

        "Hierarchy":
            hierarchy,

        **biological
    }


# ============================================================
# OUTPUT / RESUME
# ============================================================

def make_result_key(
    filename,
    environment,
    dimension,
    model,
    seed
):
    return (
        str(filename),
        str(environment),
        int(dimension),
        str(model),
        int(seed)
    )


def get_existing_keys(
    output_path
):
    """
    Read the existing output CSV and return
    all completed calculation keys.
    """

    if not os.path.exists(
        output_path
    ):
        return set()

    try:

        dataframe = pd.read_csv(
            output_path,
            usecols=[
                "file",
                "environment",
                "dimension",
                "model",
                "seed"
            ]
        )

    except Exception as error:

        print(
            f"WARNING: Could not read existing "
            f"output file '{output_path}': "
            f"{error}"
        )

        print(
            "WARNING: No existing rows will "
            "be assumed complete."
        )

        return set()

    existing_keys = set()

    for _, row in dataframe.iterrows():

        try:

            key = make_result_key(
                row["file"],
                row["environment"],
                row["dimension"],
                row["model"],
                row["seed"]
            )

            existing_keys.add(
                key
            )

        except Exception:
            continue

    return existing_keys


# ============================================================
# NEW: FAST FILE-LEVEL COMPLETION CHECK
# ============================================================

def get_completed_keys_for_file(
    filename,
    existing_keys
):
    """
    Return only the completed calculation keys
    belonging to one file.
    """

    return {
        key
        for key in existing_keys
        if key[0] == filename
    }


def file_has_all_expected_calculations(
    filename,
    existing_keys
):
    """
    FAST CHECK.

    This happens BEFORE the RSML file is parsed
    and BEFORE any environment graph is built.

    If every theoretically possible combination
    is already in TRY.csv, the entire file can be
    skipped immediately.
    """

    completed = (
        get_completed_keys_for_file(
            filename,
            existing_keys
        )
    )

    expected = set()

    for environment in (
        ENVIRONMENTAL_CONDITIONS
    ):

        for dimension in (
            EMBEDDING_DIMENSIONS
        ):

            for seed in SEEDS:

                for model in MODELS:

                    expected.add(
                        make_result_key(
                            filename,
                            environment,
                            dimension,
                            model,
                            seed
                        )
                    )

    return expected.issubset(
        completed
    )


# ============================================================
# APPEND RESULT
# ============================================================

def append_result(
    result,
    output_path
):
    """
    Append exactly one completed result.

    Existing rows are never deleted or overwritten.
    """

    directory = os.path.dirname(
        output_path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )

    fieldnames = [
        "file",
        "environment",
        "theta",
        "model",
        "dimension",
        "seed",
        "MRE",
        "Raw_MRE",
        "Stress",
        "Hierarchy",
        "Physical_Hydraulic",
        "Hydraulic",
        "Hydraulic_Preservation",
        "Embedding_Transport_Preservation",
        "Nutrient_Uptake",
        "Nutrient_Final",
        "Nutrient_Flux",
        "curvature",
        "nodes",
        "edges",
        "symbiotic_interfaces",
        "amf_species"
    ]

    file_exists = os.path.exists(
        output_path
    )

    with open(
        output_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        if not file_exists:

            writer.writeheader()

        writer.writerow({
            key:
                result.get(
                    key,
                    np.nan
                )
            for key in fieldnames
        })

        csv_file.flush()


# ============================================================
# PROCESSING
# ============================================================

def process_single_file(
    fpath,
    existing_keys,
    output_path,
    amf_nodes,
    amf_edges
):
    filename = os.path.basename(
        fpath
    )

    # ========================================================
    # CRITICAL FAST FILE-LEVEL SKIP
    # ========================================================
    #
    # THIS IS BEFORE:
    #
    #   parse_rsml_file()
    #   environmental_architecture_scaling()
    #   build_bio_hydraulic_graph()
    #   build_amf_graph()
    #   attach_amf_to_roots()
    #   coarsening
    #   Dijkstra
    #
    # Therefore a completely finished file costs essentially
    # only a CSV lookup and is immediately skipped.
    # ========================================================

    if file_has_all_expected_calculations(
        filename,
        existing_keys
    ):

        print(
            f"\n"
            f"==================================================\n"
            f"SKIP FILE — ALREADY COMPLETED\n"
            f"{filename}\n"
            f"=================================================="
        )

        return 0

    # ========================================================
    # ONLY REACH HERE IF SOMETHING IS MISSING
    # ========================================================

    print(
        f"\nPROCESSING: {filename}"
    )

    roots_data = parse_rsml_file(
        fpath
    )

    if not roots_data:

        print(
            "No roots found."
        )

        return 0

    print(
        f"Roots: {len(roots_data)} | "
        f"Original nodes: "
        f"{sum(len(r['points']) for r in roots_data)}"
    )

    saved_count = 0
    skipped_count = 0

    # --------------------------------------------------------
    # Determine exactly which calculations are missing.
    # --------------------------------------------------------

    completed_for_file = (
        get_completed_keys_for_file(
            filename,
            existing_keys
        )
    )

    for environment, theta in (
        ENVIRONMENTAL_CONDITIONS.items()
    ):

        # ----------------------------------------------------
        # BEFORE BUILDING THE ENVIRONMENT GRAPH:
        #
        # Check whether ALL calculations for this environment
        # already exist.
        # ----------------------------------------------------

        environment_complete = True

        for dimension in (
            EMBEDDING_DIMENSIONS
        ):

            for seed in SEEDS:

                for model in MODELS:

                    key = make_result_key(
                        filename,
                        environment,
                        dimension,
                        model,
                        seed
                    )

                    if key not in (
                        completed_for_file
                    ):

                        environment_complete = (
                            False
                        )

                        break

                if not environment_complete:
                    break

            if not environment_complete:
                break

        if environment_complete:

            print(
                f"Environment: {environment} "
                f"→ ALREADY COMPLETE, SKIPPING "
                f"ENTIRE ENVIRONMENT"
            )

            skipped_count += (
                len(EMBEDDING_DIMENSIONS)
                * len(SEEDS)
                * len(MODELS)
            )

            continue

        # ----------------------------------------------------
        # Environment is not complete, so NOW build graph.
        # ----------------------------------------------------

        print(
            f"Environment: {environment} "
            f"(theta={theta})"
        )

        G, D_graph, contacts = (
            prepare_graph(
                roots_data,
                amf_nodes,
                amf_edges,
                theta
            )
        )

        if G is None:

            print(
                "Skipped."
            )

            continue

        print(
            f"Coupled nodes: "
            f"{G.number_of_nodes()} | "
            f"Edges: "
            f"{G.number_of_edges()} | "
            f"Symbiotic interfaces: "
            f"{contacts}"
        )

        for dimension in (
            EMBEDDING_DIMENSIONS
        ):

            if dimension >= (
                G.number_of_nodes()
            ):

                continue

            # ------------------------------------------------
            # Dimension-level fast skip.
            # ------------------------------------------------

            dimension_complete = True

            for seed in SEEDS:

                for model in MODELS:

                    key = make_result_key(
                        filename,
                        environment,
                        dimension,
                        model,
                        seed
                    )

                    if key not in (
                        completed_for_file
                    ):

                        dimension_complete = (
                            False
                        )

                        break

                if not dimension_complete:
                    break

            if dimension_complete:

                print(
                    f"  D={dimension:2d} "
                    f"→ ALREADY COMPLETE, SKIPPING"
                )

                skipped_count += (
                    len(SEEDS)
                    * len(MODELS)
                )

                continue

            for seed in SEEDS:

                for model in MODELS:

                    key = make_result_key(
                        filename,
                        environment,
                        dimension,
                        model,
                        seed
                    )

                    # ========================================
                    # CALCULATION-LEVEL SKIP
                    # ========================================

                    if key in (
                        completed_for_file
                    ):

                        skipped_count += 1

                        print(
                            f" SKIP EXISTING | "
                            f"{model:8s} | "
                            f"D={dimension:2d} | "
                            f"Seed={seed:4d}"
                        )

                        continue

                    set_seed(
                        seed
                    )

                    try:

                        # ====================================
                        # RUN MODEL
                        # ====================================

                        if model == "Poincare":

                            (
                                embedding,
                                D_pred,
                                curvature
                            ) = run_poincare(
                                D_graph,
                                dimension,
                                seed,
                                POINCARE_EPOCHS
                            )

                            is_poincare = True

                        elif model == "MDS":

                            (
                                embedding,
                                D_pred
                            ) = run_mds(
                                D_graph,
                                dimension,
                                seed
                            )

                            curvature = np.nan

                            is_poincare = False

                        else:

                            (
                                embedding,
                                D_pred
                            ) = run_spectral(
                                D_graph,
                                dimension,
                                seed
                            )

                            curvature = np.nan

                            is_poincare = False

                        # ====================================
                        # METRICS
                        # ====================================

                        metrics = compute_metrics(
                            G,
                            embedding,
                            D_graph,
                            D_pred,
                            theta,
                            is_poincare=
                                is_poincare,
                            poincare_c=(
                                curvature
                                if is_poincare
                                else None
                            )
                        )

                        # ====================================
                        # RESULT
                        # ====================================

                        result = {

                            "file":
                                filename,

                            "environment":
                                environment,

                            "theta":
                                theta,

                            "model":
                                model,

                            "dimension":
                                dimension,

                            "seed":
                                seed,

                            "MRE":
                                metrics["MRE"],

                            "Raw_MRE":
                                metrics["Raw_MRE"],

                            "Stress":
                                metrics["Stress"],

                            "Hierarchy":
                                metrics["Hierarchy"],

                            "Physical_Hydraulic":
                                metrics[
                                    "Physical_Hydraulic"
                                ],

                            "Hydraulic":
                                metrics[
                                    "Hydraulic"
                                ],

                            "Hydraulic_Preservation":
                                metrics[
                                    "Hydraulic_Preservation"
                                ],

                            "Embedding_Transport_Preservation":
                                metrics[
                                    "Embedding_Transport_Preservation"
                                ],

                            "Nutrient_Uptake":
                                metrics[
                                    "Nutrient_Uptake"
                                ],

                            "Nutrient_Final":
                                metrics[
                                    "Nutrient_Final"
                                ],

                            "Nutrient_Flux":
                                metrics[
                                    "Nutrient_Flux"
                                ],

                            "curvature":
                                curvature,

                            "nodes":
                                G.number_of_nodes(),

                            "edges":
                                G.number_of_edges(),

                            "symbiotic_interfaces":
                                contacts,

                            "amf_species":
                                (
                                    f"{AMF_NAME} "
                                    f"({AMF_MODERN_NAME})"
                                )
                        }

                        # ====================================
                        # SAVE IMMEDIATELY
                        # ====================================

                        append_result(
                            result,
                            output_path
                        )

                        existing_keys.add(
                            key
                        )

                        completed_for_file.add(
                            key
                        )

                        saved_count += 1

                        print(
                            f" {model:8s} | "
                            f"D={dimension:2d} | "
                            f"Seed={seed:4d} | "
                            f"MRE={metrics['MRE']:.6f} | "
                            f"Stress={metrics['Stress']:.6f} | "
                            f"Hierarchy={metrics['Hierarchy']:.6f} | "
                            f"Hydraulic={metrics['Hydraulic']:.6f} | "
                            f"TransportPres="
                            f"{metrics['Embedding_Transport_Preservation']:.6f} | "
                            f"Nutrient="
                            f"{metrics['Nutrient_Uptake']:.6f}"
                        )

                    except Exception as error:

                        print(
                            f" {model} failed: "
                            f"{error}"
                        )

                gc.collect()

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

    print(
        f"Finished {filename}: "
        f"{saved_count} new rows, "
        f"{skipped_count} existing rows skipped."
    )

    return saved_count


# ============================================================
# FULL EXPERIMENT SUITE
# ============================================================

def run_full_experiment_suite(
    dataset_dir,
    amf_nodes_csv=AMF_NODES_CSV,
    amf_edges_csv=AMF_EDGES_CSV,
    output_path=OUTPUT_CSV_PATH,
    max_files=None,
    selected_files=None,
    clear_output=False
):
    """
    Run the complete experiment.

    RESUME BEHAVIOR
    ---------------

    clear_output=False:

        Existing TRY.csv is preserved.

        Before processing each file, the program checks
        whether all expected calculations for that file
        already exist.

        If YES:
            The file is skipped immediately.

        No RSML parsing.
        No graph construction.
        No Dijkstra.
        No embeddings.

        If the file is incomplete:
            Only missing environments/calculations are run.

    clear_output=True:

        Existing TRY.csv is explicitly deleted.
    """

    if not os.path.isdir(
        dataset_dir
    ):

        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{dataset_dir}"
        )

    amf_nodes, amf_edges = (
        load_amf_inputs(
            amf_nodes_csv,
            amf_edges_csv
        )
    )

    files = sorted([
        os.path.join(
            dataset_dir,
            filename
        )
        for filename in os.listdir(
            dataset_dir
        )
        if filename.lower().endswith(
            (
                ".rsml",
                ".xml"
            )
        )
    ])

    if selected_files is not None:

        selected = set(
            selected_files
        )

        files = [
            p
            for p in files
            if os.path.basename(p)
            in selected
        ]

    elif max_files is not None:

        files = files[
            :max_files
        ]

    if not files:

        print(
            "No RSML/XML files found."
        )

        return

    # ========================================================
    # OUTPUT SAFETY
    # ========================================================

    if clear_output:

        if os.path.exists(
            output_path
        ):

            print(
                "\nWARNING: clear_output=True"
            )

            print(
                f"Deleting existing output file:\n"
                f"{output_path}"
            )

            os.remove(
                output_path
            )

    else:

        if os.path.exists(
            output_path
        ):

            print(
                "\nExisting output file detected."
            )

            print(
                "It will NOT be deleted."
            )

            print(
                "Completed files/calculations "
                "will be skipped."
            )

    # ========================================================
    # LOAD EXISTING RESULTS ONCE
    # ========================================================

    existing_keys = (
        get_existing_keys(
            output_path
        )
    )

    print(
        "=" * 90
    )

    print(
        "COUPLED ZEA MAYS - AMF "
        "SYMBIOTIC EMBEDDING EXPERIMENT"
    )

    print(
        "(FAST FILE-LEVEL RESUME)"
    )

    print(
        "=" * 90
    )

    print(
        f"Dataset: {dataset_dir}"
    )

    print(
        f"Files selected: {len(files)}"
    )

    print(
        f"AMF: {AMF_NAME} "
        f"(modern: {AMF_MODERN_NAME})"
    )

    print(
        f"AMF nodes: {len(amf_nodes)}"
    )

    print(
        f"AMF edges: {len(amf_edges)}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Existing completed rows: "
        f"{len(existing_keys)}"
    )

    print(
        f"Output deletion enabled: "
        f"{clear_output}"
    )

    print(
        "=" * 90
    )

    total_saved = 0

    try:

        for index, filepath in enumerate(
            files,
            start=1
        ):

            filename = os.path.basename(
                filepath
            )

            print(
                f"\n[{index}/{len(files)}] "
                f"{filename}"
            )

            # =================================================
            # ULTRA-EARLY FILE CHECK
            # =================================================
            #
            # This happens BEFORE process_single_file()
            # does anything expensive.
            # =================================================

            if file_has_all_expected_calculations(
                filename,
                existing_keys
            ):

                print(
                    f"*** FAST SKIP: {filename} "
                    f"— ALL CALCULATIONS ALREADY COMPLETE ***"
                )

                continue

            saved = process_single_file(
                filepath,
                existing_keys,
                output_path,
                amf_nodes,
                amf_edges
            )

            total_saved += saved

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    except KeyboardInterrupt:

        print(
            "\nPROCESS INTERRUPTED."
        )

        print(
            "Completed rows have already "
            "been saved."
        )

        print(
            "You can safely rerun the script."
        )

        print(
            "Completed files/calculations "
            "will be skipped."
        )

        return

    print(
        "\n"
        + "=" * 90
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        f"New rows saved: "
        f"{total_saved}"
    )

    print(
        f"Total completed rows now known: "
        f"{len(existing_keys)}"
    )

    print(
        f"Output: "
        f"{os.path.abspath(output_path)}"
    )

    print(
        "=" * 90
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    DIRECTORY_PATH = (
        r"C:\Users\user\Downloads\4836890"
    )

    run_full_experiment_suite(
        dataset_dir=DIRECTORY_PATH,

        max_files=None,

        clear_output=False
    )