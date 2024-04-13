import jsonlines
import mmh3
from typing import Callable, List
from multiprocessing import Pool, Manager
from tqdm import tqdm

def ngrams(text, n):
    return [text[i:i+n] for i in range(len(text)-n+1)]

def hashfunc_signed_32_from_seed(seed: int) -> Callable[[str], int]:
    return lambda text: mmh3.hash(text, seed, signed=True)

def get_minhash(tokens: List[str], hashfunc: Callable[[str], int]) -> int:
    return min([hashfunc(text) for text in tokens])

def calc_lsh(text: str, n_minhash: int = 400, n_gram: int = 5, n_buckets: int = 20, bucket_size: int = 20) -> List[str]:
    assert n_minhash == n_buckets * bucket_size
    fingerprints = []
    for seed in range(n_minhash):
        hashfunc = hashfunc_signed_32_from_seed(seed)
        minhash = get_minhash(ngrams(text, n=n_gram), hashfunc)
        fingerprints.append(minhash)

    lshs = []
    for bucket_idx in range(n_buckets):
        lshs.append(
            str(bucket_idx)
            + "+"
            + "".join(
                [
                    format(fingerprints[fp_idx], "04x")[-4:]
                    for fp_idx in range(
                        bucket_idx * bucket_size, (bucket_idx + 1) * bucket_size
                    )
                ]
            )
        )
    return lshs

def is_duplicate(item, seen):
    text = item['text']
    lshs = calc_lsh(text)
    for lsh in lshs:
        if lsh in seen:
            return True, None
        else:
            seen[lsh] = True
    return False, item

def deduplicate_batch(batch, seen, num_processes=4):
    with Pool(processes=num_processes) as pool:
        args_list = [(item, seen) for item in batch]
        results = []
        for result in pool.imap_unordered(is_duplicate_unpack, args_list):
            results.append(result)
    valid_items = [item for _, item in results if item is not None]
    return valid_items

def deduplicate_jsonl(input_file, output_file, batch_size=10000, num_processes=4):
    manager = Manager()
    seen = manager.dict()
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        batch = []
        for item in tqdm(reader, desc="Reading data"):
            batch.append(item)
            if len(batch) == batch_size:
                valid_items = deduplicate_batch(batch, seen, num_processes)
                writer.write_all(valid_items)
                batch = []
        if batch:
            valid_items = deduplicate_batch(batch, seen, num_processes)
            writer.write_all(valid_items)

def is_duplicate_unpack(args):
    return is_duplicate(*args)

if __name__ == '__main__':
    deduplicate_jsonl('filtered_oscar.jsonl', 'deduplicated_oscar.jsonl', batch_size=131072, num_processes=32)
    #deduplicate_jsonl('filtered_oscar.jsonl', 'deduplicated_oscar.jsonl', batch_size=16384, num_processes=32)