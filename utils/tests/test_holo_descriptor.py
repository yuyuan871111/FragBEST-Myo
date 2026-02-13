import os

import pytest


@pytest.mark.xdist_group(name="holo_descriptor")
class TestHoloDescriptor:
    from ..ppseg.holo_descriptor.holo_descriptor import HoloDescriptor

    current_dir = os.getcwd()

    ply_path = f"{current_dir}/tests/test_data/5N69_protein.ply"
    json_path = f"{current_dir}/tests/test_data/5N69_protein.json"

    holo_descriptor = HoloDescriptor(ply_path)

    def test_init(self):
        assert self.holo_descriptor is not None
        assert self.holo_descriptor.ply_path == self.ply_path

    def test_run(self):
        self.holo_descriptor.run()
        for each in self.holo_descriptor.results.keys():
            assert self.holo_descriptor.results[each] is not None

    def test_save(self):
        self.holo_descriptor.save(json_path=self.json_path)
        assert os.path.exists(self.json_path)

        os.system(f"rm {self.json_path}")


@pytest.mark.xdist_group(name="hd_analyser")
class TestHoloDescriptorAnalyser:
    from ..ppseg.holo_descriptor.holo_descriptor import HoloDescriptorAnalyser

    current_dir = os.getcwd()

    source_path = f"{current_dir}/tests/test_data/pred_json"
    frag_info_path = f"{current_dir}/tests/test_data/ligand_fragments.json"

    # TestHoloDescriptorAnalyser
    hd_analyser = HoloDescriptorAnalyser(
        source_path=source_path, frag_info_path=frag_info_path
    )

    def test_init(self):
        assert self.hd_analyser is not None
        assert self.hd_analyser.source_path == self.source_path
        assert self.hd_analyser.frag_info_path == self.frag_info_path

    def test_list_files(self):
        self.hd_analyser.list_files()
        assert len(self.hd_analyser.files) == 6

    def test_read(self):
        self.hd_analyser.read(holospace_calc=True)
        assert len(self.hd_analyser.descriptors_df) == 6
        assert len(self.hd_analyser.holospace_frag_volumes) == 6

    def test_calculate_zscore(self):
        self.hd_analyser.calculate_zscore("nonbck_ratio")
        self.hd_analyser.calculate_zscore("nonbck_class_pt_ratio")
        self.hd_analyser.calculate_zscore("holospace_frag_score")
        self.hd_analyser.calculate_zscore("overall_predprobs")

        assert "nonbck_ratio_zscore" in self.hd_analyser.descriptors_df.columns
        assert "nonbck_class_pt_ratio_zscore" in self.hd_analyser.descriptors_df.columns
        assert "holospace_frag_score_zscore" in self.hd_analyser.descriptors_df.columns
        assert "overall_predprobs_zscore" in self.hd_analyser.descriptors_df.columns

        self.hd_analyser.calculate_zscore("num_of_classes")
        assert "num_of_classes_zscore" not in self.hd_analyser.descriptors_df.columns

    def test_set_rank(self):
        zscore_columns = [
            "nonbck_ratio_zscore",
            "nonbck_class_pt_ratio_zscore",
            "overall_predprobs_zscore",
            "holospace_frag_score_zscore",
        ]
        weights = [1, 1, 1, 1]

        self.hd_analyser.set_rank(
            filter_warning=False, zscore_columns=zscore_columns, weights=weights
        )
        assert "overall_score" in self.hd_analyser.descriptors_df.columns
        assert "rank" in self.hd_analyser.descriptors_df.columns
