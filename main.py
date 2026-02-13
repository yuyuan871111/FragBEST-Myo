"""Modified by Yu-Yuab (Stuart) Yang - 2024
from ignite generator template - 2024

"""

import ignite.distributed as idist

from utils.ppseg.ignite.utils import setup_config


# main entrypoint
def main():
    config = setup_config()

    match config.mode:
        case "train":
            from utils.ppseg.train import run_training

            with idist.Parallel(config.backend) as p:
                p.run(run_training, config=config)

        case "test":
            from utils.ppseg.test import run_testing

            with idist.Parallel(config.backend) as p:
                p.run(run_testing, config=config)

        case "dataprep_PPSCM_OMB":
            from utils.datasets.PPSCM_OMB import run_preparation

            run_preparation(config)

        case "dataprep_PPSCM_Apo":
            from utils.datasets.PPSCM_Apo import run_preparation

            run_preparation(config)

        case "dataprep_PRCM_OMB":
            from utils.datasets.PRCM_OMB import run_preparation

            run_preparation(config)

        case "dataprep_PRCM_Apo":
            from utils.datasets.PRCM_Apo import run_preparation

            run_preparation(config)

        case "dataprep_PR_to_PPS_Apo":
            from utils.datasets.PR_to_PPS_Apo import run_preparation

            run_preparation(config)

        case "datacheck_PRCM_OMB":
            from utils.datasets.PRCM_check import run_checking

            run_checking(config)

        case "datacheck_PPSCM_OMB" | "datacheck_PPSCM_Apo":
            from utils.datasets.PPSCM_check import run_checking

            run_checking(config)

        case "predict" | "predict_PPSCM" | "predict_PRCM" | "predict_PR_to_PPS":
            from utils.datasets.general import run_prediction

            run_prediction(config)

        case _:
            raise NotImplementedError("Invalid mode")


if __name__ == "__main__":
    main()
