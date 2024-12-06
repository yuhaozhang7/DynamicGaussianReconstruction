import scene
import random
import torch
import copy


def sample_small_dataset(scene, num=10, size=40, diff=0, reverse=False):

    all_cameras = scene.getTrainCameras().copy()
    all_cameras.sort(key=lambda cam: cam.fid.item())
    len_all_cameras = len(all_cameras)
    assert len_all_cameras >= size, "original dataset must be greater than extracted dataset"

    sampled_datasets = []
    for i in range(num):
        largest_diff = len_all_cameras // size
        if diff == 0:
            diff = random.randint(1, largest_diff)
        largest_start_idx = len_all_cameras - (diff * (size - 1)) - 1
        start_idx = random.randint(0, largest_start_idx)

        sampled_cameras = []
        for j in range(size):
            sampled_cameras.append(copy.deepcopy(all_cameras[start_idx + j * diff]))

        if i % 2 == 0 and reverse:
            sampled_cameras.reverse()
        
        interval = 1 / (size - 1) if size > 1 else 1
        for idx, cam in enumerate(sampled_cameras):
            cam.fid = torch.tensor([idx * interval], device=cam.fid.device)

        sampled_datasets.append(sampled_cameras)

    # test
    for dataset in sampled_datasets:
        names = [cam.image_name for cam in dataset]
        print(names)

    return sampled_datasets
