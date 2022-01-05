from training import dataset
import numpy as np
# import rich

# def test_dataset():
#     ds = dataset.CoordImageDataset("/data/cyclegan/horse2zebra/testA")
#     rich.inspect(ds)

#     assert ds.resolution == 256

#     item = ds[0]
#     rich.inspect(item)

#     assert item[0].shape == (5, 256, 256)

#     assert all(x == 0 for x in item[0][3, :, 0])
#     assert all(x == 255 for x in item[0][3, :, -1])

#     assert all(x == 0 for x in item[0][4, 0, :])
#     assert all(x == 255 for x in item[0][4, -1, :])

def test_np_dataset(tmp_path):
    np_path = tmp_path/"data.npy"
    n = 100
    res = 64
    data = np.ones((n, 3, res, res), np.float32)
    np.save(np_path, data)

    ds = dataset.NpArrayDataset(np_path)
    assert len(ds) == n
    assert ds[0][0].shape == (3, res, res)
