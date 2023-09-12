from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset, DataLoader
import math


class MyIterableDataset(IterableDataset):

    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))


ds = MyIterableDataset(start=3, end=7)  # [3, 4, 5, 6]

# 单进程
print(list(DataLoader(ds, num_workers=0)))
# [tensor([3]), tensor([4]), tensor([5]), tensor([6])]


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) /
                     float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


# 如果num_workers的值大于0，要在运行的部分放进__main__()函数里，才不会有错
if __name__ == '__main__':

    # 开启多进程（num_workers > 0）
    print(list(DataLoader(ds, num_workers=2)))
    # [tensor([3]), tensor([3]), tensor([4]), tensor([4]), tensor([5]), tensor([5]), tensor([6]), tensor([6])]

    # 如果在 DataLoader 中开启多进程（num_workers > 0），那么在加载迭代型数据集时必须进行专门的设置，否则会重复访问样本
    print(list(DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
    # [tensor([3]), tensor([5]), tensor([4]), tensor([6])]
