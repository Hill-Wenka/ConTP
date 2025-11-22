import concurrent.futures
import multiprocessing as mp
from tqdm.notebook import tqdm


def concurrent_submit(parallel_func, params, cpu_num=mp.cpu_count(), desc=None, threading=False):
    # params: list of tuple, e.g. [(a1, b1), (a2, b2), ...]
    pbar = tqdm(total=len(params))
    desc = desc if desc is not None else 'Parallel Running, cpu_num: %d' % cpu_num
    pbar.set_description(desc)

    if threading:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=cpu_num)  # 适合I/O密集型任务
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=cpu_num)  # 适合CPU密集型任务

    with executor as executor:
        futures = [executor.submit(parallel_func, *param) for param in params]
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print('concurrent submit Exception: ', e)
                results.append(None)
            pbar.update(1)
    return results
