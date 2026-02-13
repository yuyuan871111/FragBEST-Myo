import os

LIGAND_FRAG_INFO_PATH = os.path.join(
    os.path.dirname(__file__), "ligand_fragments_example.json"
)

HOLO_DESCRIPTOR_PRESETS = {
    # from apo PPS trajectory (exclude outliers)
    "pps": {
        "overall_predprobs": {"mean": 0.9907177424460778, "std": 0.0031131054149345846},
        "nonbck_ratio": {"mean": 0.10094656870699203, "std": 0.060613787423331365},
        "nonbck_class_pt_ratio": {
            "mean": 0.018897312735291327,
            "std": 0.009324337878677633,
        },
        "num_of_classes": {"mean": 6.037214502475643, "std": 1.253273265346901},
        "num_interest_points": {"mean": 1287.0130969493691, "std": 71.67864143339092},
        "holospace_volume": {"mean": 966.3776276168828, "std": 561.9693337796247},
        "holospace_frag_score": {"mean": 0.6511649471909706, "std": 0.2716907001392197},
    },
    # from apo PR trajectory
    "pr": {
        "overall_predprobs": {"mean": 0.9899470306697108, "std": 0.0027646117234690947},
        "nonbck_ratio": {"mean": 0.1587055159348904, "std": 0.056228165797437904},
        "nonbck_class_pt_ratio": {
            "mean": 0.0289937529720443,
            "std": 0.007864030521360485,
        },
        "num_of_classes": {"mean": 6.39802658403327, "std": 0.9407600873471021},
        "num_interest_points": {"mean": 1083.6912664111555, "std": 62.25896738059542},
        "holospace_volume": {"mean": 1165.7606369941282, "std": 400.7451164853268},
        "holospace_frag_score": {
            "mean": 0.7420283974754991,
            "std": 0.17409888485456923,
        },
    },
}
