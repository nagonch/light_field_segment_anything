from data import HCIOldDataset
from LF_SAM import get_sam


def lawnmower(LF):
    s, t, u, v, c = LF.shape
    frame_num = 0
    index_sequence = []
    for i in range(s):
        if i % 2 == 0:
            for j in range(t):
                frame_num += 1
                index_sequence.append((i, j))
        else:
            for j in range(t - 1, -1, -1):
                frame_num += 1
                index_sequence.append((i, j))
    num_inds = len(index_sequence)
    ind_sequence_up = index_sequence[: num_inds // 2][::-1]
    ind_sequence_down = index_sequence[num_inds // 2 + 1 :]
    return ind_sequence_up, ind_sequence_down


def xmem_LF(LF):
    s, t, u, v, c = LF.shape
    central_subap = LF[s // 2, t // 2]
    sam = get_sam()
    masks = sam.generator.generate(central_subap)
    return masks


if __name__ == "__main__":
    dataset = HCIOldDataset()
    LF, _, _ = dataset[0]
    xmem_LF(LF)
    # ind_pattern = lawnmower(LF)
    # print(ind_pattern[len(ind_pattern) // 2])
    # # print(type(LF))
