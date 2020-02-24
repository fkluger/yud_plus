import glob
import os
import numpy as np
import scipy.io
import imageio
from torch.utils.data import Dataset
import lsd


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class YUDVP:

    def __init__(self, data_dir_path, split='', keep_in_mem=True, normalize_coords=False,
                 return_images=False, extract_lines=False):
        self.data_dir = data_dir_path

        self.image_folders = glob.glob(os.path.join(self.data_dir, "P*/"))
        self.image_folders.sort()

        self.keep_in_mem = keep_in_mem
        self.normalize_coords = normalize_coords
        self.return_images = return_images

        if split is not None:
            if split == "train":
                self.set_ids = list(range(0, 25))
            elif split == "test":
                self.set_ids = list(range(25, 102))
            elif split == "all":
                self.set_ids = list(range(0, 102))
            else:
                assert False, "invalid split"

        self.dataset = [None for _ in self.set_ids]

        camera_params = scipy.io.loadmat(os.path.join(self.data_dir, "cameraParameters.mat"))

        f = camera_params['focal'][0, 0]
        ps = camera_params['pixelSize'][0, 0]
        pp = camera_params['pp'][0, :]

        self.K = np.matrix([[f / ps, 0, pp[0]], [0, f / ps, pp[1]], [0, 0, 1]])
        self.S = np.matrix([[2.0 / 640, 0, -1], [0, 2.0 / 640, -0.75], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.S * self.K) if normalize_coords else np.linalg.inv(self.K)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):

        id = self.set_ids[key]

        datum = self.dataset[key]

        if datum is None:
            image_path = glob.glob(os.path.join(self.image_folders[id], "P*.jpg"))[0]
            image_rgb = imageio.imread(image_path)
            image = rgb2gray(image_rgb)

            lsd_line_segments = lsd.detect_line_segments(image)

            if self.normalize_coords:
                lsd_line_segments[:, 0] -= 320
                lsd_line_segments[:, 2] -= 320
                lsd_line_segments[:, 1] -= 240
                lsd_line_segments[:, 3] -= 240
                lsd_line_segments[:, 0:4] /= 320.

            line_segments = np.zeros((lsd_line_segments.shape[0], 7 + 2 + 3 + 3))
            for li in range(line_segments.shape[0]):
                p1 = np.array([lsd_line_segments[li, 0], lsd_line_segments[li, 1], 1])
                p2 = np.array([lsd_line_segments[li, 2], lsd_line_segments[li, 3], 1])
                centroid = 0.5 * (p1 + p2)
                line = np.cross(p1, p2)
                line /= np.linalg.norm(line[0:2])
                line_segments[li, 0:3] = p1
                line_segments[li, 3:6] = p2
                line_segments[li, 6:9] = line
                line_segments[li, 9:12] = centroid
                line_segments[li, 12:15] = lsd_line_segments[li, 4:7]

            mat_gt_path = glob.glob(os.path.join(self.image_folders[id], "P*GroundTruthVP_CamParams.mat"))[0]
            gt_data = scipy.io.loadmat(mat_gt_path)

            true_vds = np.matrix(gt_data['vp'])
            true_vds[1, :] *= -1

            true_vps = self.K * true_vds

            num_vp = true_vps.shape[1]
            tvp_list = []

            for vi in range(num_vp):
                true_vps[:, vi] /= true_vps[2, vi]

                tVP = np.array(true_vps[:, vi])[:, 0]
                tVP /= tVP[2]

                tvp_list += [tVP]

            true_vps = np.vstack(tvp_list)

            vps = true_vps

            datum = {'line_segments': line_segments, 'VPs': vps, 'id': id, 'VDs': true_vds}

            if self.return_images:
                datum['image'] = np.array(image_rgb)

            for vi in range(datum['VPs'].shape[0]):
                if self.normalize_coords:
                    datum['VPs'][vi, :][0] -= 320
                    datum['VPs'][vi, :][1] -= 240
                    datum['VPs'][vi, :][0:2] /= 320.
                datum['VPs'][vi, :] /= np.linalg.norm(datum['VPs'][vi, :])

            if self.keep_in_mem:
                self.dataset[key] = datum

        return datum


class YUDVPDataset(Dataset):

    def __init__(self, data_dir_path, max_num_segments, max_num_vps, split='train', keep_in_mem=True,
                 mat_file_path=None, permute_lines=True, return_images=False):
        self.dataset = YUDVP(data_dir_path, split, keep_in_mem, normalize_coords=True, return_images=return_images)
        self.max_num_segments = max_num_segments
        self.max_num_vps = max_num_vps
        self.permute_lines = permute_lines
        self.return_images = return_images

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        datum = self.dataset[key]

        if self.max_num_segments is None:
            max_num_segments = datum['line_segments'].shape[0]
        else:
            max_num_segments = self.max_num_segments

        line_segments = np.zeros((max_num_segments, 15)).astype(np.float32)
        vps = np.zeros((self.max_num_vps, 3)).astype(np.float32)
        mask = np.zeros((max_num_segments,)).astype(np.int)

        num_actual_line_segments = np.minimum(datum['line_segments'].shape[0], max_num_segments)
        if self.permute_lines:
            np.random.shuffle(line_segments)
        line_segments[0:num_actual_line_segments, :] = datum['line_segments'][0:num_actual_line_segments, :]

        mask[0:num_actual_line_segments] = 1

        num_actual_vps = np.minimum(datum['VPs'].shape[0], self.max_num_vps)
        vps[0:num_actual_vps, :] = datum['VPs'][0:num_actual_vps]
        if self.return_images:
            return line_segments, vps, num_actual_line_segments, num_actual_vps, mask, datum['image']
        else:
            return line_segments, vps, num_actual_line_segments, num_actual_vps, mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sampling

    dataset = YUDVP("/tnt/data/scene_understanding/YUD_relabelled",
                    split='all', normalize_coords=False, return_images=True)

    max_num_vp = 0
    max_num_ls = 0
    all_distances_smallest = []
    all_distances_second = []
    all_num_vps = []
    for idx in range(len(dataset)):
        vps = dataset[idx]['VPs']
        num_vps = vps.shape[0]
        print("%d vp: " % idx, num_vps)
        all_num_vps += [num_vps]
        if num_vps > max_num_vp: max_num_vp = num_vps
        num_ls = dataset[idx]['line_segments'].shape[0]
        if num_ls > max_num_ls: max_num_ls = num_ls

        ls = dataset[idx]['line_segments']
        vp = dataset[idx]['VPs']

        distances_per_img = []

        for vi in range(vp.shape[0]):
            distances = sampling.vp_consistency_measure_angle_np(vp[vi], ls)
            distances_per_img += [distances]
        distances_per_img = np.sort(np.vstack(distances_per_img), axis=0)

        smallest = distances_per_img[0, :]
        all_distances_smallest += [smallest]

    print("max vps: ", max_num_vp)
    print(np.unique(all_num_vps, return_counts=True))

    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(9, 3))
    values, bins, patches = plt.hist(all_num_vps, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    print(values)
    print(bins)
    plt.show()

    all_distances_smallest = np.hstack(all_distances_smallest)

    bins = np.linspace(0, .1, 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(all_distances_smallest, bins)
    ax.set_yscale("log")
    plt.show()

