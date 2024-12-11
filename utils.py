import contextlib
import sys
from tqdm.contrib import DummyTqdmFile
from functools import wraps
from typing import List
import pandas as pd
import wandb 
import ipdb

@contextlib.contextmanager
def stream_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        print("## stream_redirect_tqdm-try")
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        print("## stream_redirect_tqdm-except")
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        print("## stream_redirect_tqdm-finally")
        sys.stdout, sys.stderr = orig_out_err


def wandbLogger(func):
    '''
    wandb装饰器
    主函数返回DataFrame，记录每个epoch的metric
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        result_dfs = None
        for i, result_df in enumerate(func(*args, **kwargs)):
            result_df.columns = result_df.columns.droplevel(0)
            loginfo = result_df.to_dict('records')[0]
            loginfo['epoch'] = i
            wandb.log(loginfo)
            if result_dfs is None:
                result_dfs = result_df
            else:
                result_dfs = pd.concat([result_dfs, result_df], axis=0)
        for col in result_dfs.columns:
            wandb.log({col+'_max' : result_dfs[col].max()})
    return wrapper
            
