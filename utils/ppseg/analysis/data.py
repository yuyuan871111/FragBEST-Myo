from natsort import natsorted


def md_source_counter(filename_list):
    filename_list = natsorted(filename_list)
    counts = {}
    for filename in filename_list:
        filename_ = filename.replace(".h5", "").split("_")
        md_from = filename_[1]
        no = filename_[-1]
        if md_from not in counts:
            counts[md_from] = []
        counts[md_from].append(no)
    return counts
