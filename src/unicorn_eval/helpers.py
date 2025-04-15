import multiprocessing
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Process
from pathlib import Path

import psutil


class PredictionProcessingError(Exception):
    pass


def display_processing_report(succeeded, canceled, failed):
    print("PROCESSING REPORT")
    total = len(succeeded) + len(canceled) + len(failed)

    print(f"Succeeded ({len(succeeded)}/{total}):")
    for s in succeeded or "-":
        print(f"\t{s}")
    print(f"Canceled ({len(canceled)}/{total}):")
    for c in canceled or "-":
        print(f"\t{c}")
    print(f"Failed ({len(failed)}/{total}):")
    for f in failed or "-":
        print(f"\t{f}")
    print("")


def get_max_workers():
    """
    Returns the maximum number of concurrent workers

    The optimal number of workers ultimately depends on how many resources
    each process will call upon.

    To limit this, update the Dockerfile GRAND_CHALLENGE_MAX_WORKERS
    """

    environ_cpu_limit = os.getenv("GRAND_CHALLENGE_MAX_WORKERS")
    cpu_count = multiprocessing.cpu_count()
    return min(
        [
            int(environ_cpu_limit or cpu_count),
            cpu_count,
        ]
    )


def run_prediction_processing(*, fn, predictions):
    """
    Processes predictions in a separate process.

    This takes child processes into account:
    - if any child process is terminated, all prediction processing will abort
    - after prediction processing is done, all child processes are terminated

    Note that the results are returned in completing order.

    Parameters
    ----------
    fn : function
        Function to execute that will process each prediction

    predictions : list
        List of predictions.

    Returns
    -------
    A list of results
    """
    with Manager() as manager:
        results = manager.dict()
        errors = manager.dict()

        pool_worker = _start_pool_worker(
            fn=fn,
            predictions=predictions,
            max_workers=get_max_workers(),
            results=results,
            errors=errors,
        )
        try:
            pool_worker.join()
        finally:
            pool_worker.terminate()

        failed = set(errors.keys())
        succeeded = set(results.keys())
        canceled = set(p["pk"] for p in predictions) - (failed | succeeded)

        display_processing_report(succeeded, canceled, failed)

        if errors:
            for prediction_pk, tb_str in errors.items():
                print(
                    f"Error in prediction: {prediction_pk}\n{tb_str}",
                    file=sys.stderr,
                )

            raise PredictionProcessingError()

        return list(results.values())


def _start_pool_worker(fn, predictions, max_workers, results, errors):
    process = Process(
        target=_pool_worker,
        name="PredictionProcessing",
        kwargs=dict(
            fn=fn,
            predictions=predictions,
            max_workers=max_workers,
            results=results,
            errors=errors,
        ),
    )
    process.start()

    return process


def _pool_worker(*, fn, predictions, max_workers, results, errors):
    caught_exception = False
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        try:
            # Submit the processing tasks of the predictions
            futures = [executor.submit(fn, prediction) for prediction in predictions]
            future_to_predictions = {
                future: item for future, item in zip(futures, predictions, strict=True)
            }

            for future in as_completed(future_to_predictions):
                prediction = future_to_predictions[future]
                prediction_pk = prediction["pk"]

                error = future.exception()

                if error:
                    # Cannot pickle tracestacks, so format it here
                    tb_exception = traceback.TracebackException.from_exception(error)
                    errors[prediction_pk] = "".join(tb_exception.format())

                    if not caught_exception:  # Hard stop
                        caught_exception = True

                        executor.shutdown(wait=False, cancel_futures=True)
                        _terminate_child_processes()
                else:
                    results[prediction_pk] = future.result()

        finally:
            # Be aggresive in cleaning up any left-over processes
            _terminate_child_processes()


def _terminate_child_processes():
    process = psutil.Process(os.getpid())
    children = process.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass  # Not a problem

    # Wait for processes to terminate
    _, still_alive = psutil.wait_procs(children, timeout=5)

    # Forcefully kill any remaining processes
    for p in still_alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass  # That is fine

    # Finally, prevent zombies by waiting for all child processes
    try:
        os.waitpid(-1, 0)
    except ChildProcessError:
        pass  # No child processes, that if fine


def tree(dir_path: Path, prefix: str = ""):
    """A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    """
    space = "    "
    branch = "│   "
    # pointers:
    tee = "├── "
    last = "└── "

    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents, strict=True):
        yield prefix + pointer + path.name
        if path.is_dir():  # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix + extension)
