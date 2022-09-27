"""Set GPU visibility before importing MPI and torch when possible"""
import os


def set_visibility(local_rank: int) -> int:
    """Update CUDA_VISIBLE_DEVICES to select one GPU based on `local_rank` of process.
    Return the device number of the selected GPU, and -1 if no GPUs specified.
    (Note that CUDA_VISIBLE_DEVICES must be set explicitly to use GPUs.)"""
    cuda_dev_str = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_dev_str:
        # Select one GPU and make sure it's only one visible to torch:
        cuda_devs = [int(s) for s in cuda_dev_str.split(",")]
        cuda_dev_selected = cuda_devs[local_rank % len(cuda_devs)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_dev_selected)
        return cuda_dev_selected
    else:
        # Disable GPUs unless explicitly requested:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return -1


# Process GPU visibility in environment BEFORE torch and MPI imports
for local_rank_key in ("OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"):
    if local_rank_str := os.environ.get(local_rank_key):
        set_visibility(int(local_rank_str))
        break
