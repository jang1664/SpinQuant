import time
from collections import defaultdict
from contextlib import contextmanager
import threading
from functools import wraps
import pandas as pd
from matplotlib import pyplot as plt
import torch

class Profiler:
  _instance = None
  _lock = threading.Lock()

  @staticmethod
  def get_instance():
    if Profiler._instance is None:
      with Profiler._lock:
        if Profiler._instance is None:
          Profiler._instance = Profiler()
    return Profiler._instance

  def __init__(self):
    self.records = defaultdict(lambda: {
        "call_count": 0,
        "total_time": 0.0,
        "cum_time": 0.0
    })
    self.stack = []
    self.enabled = True  # Profiling enabled by default

  def enable(self):
    self.enabled = True

  def disable(self):
    self.enabled = False

  def is_enabled(self):
    return self.enabled

  def get_parent(self, tag):
    parts = tag.split('.')
    return '.'.join(parts[:-1]) if len(parts) > 1 else None

  @contextmanager
  def measure(self, tag):
    profiler = get_profiler()
    if not profiler.enabled:
      yield
      return
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    parent = profiler.stack[-1] if profiler.stack else None
    profiler.stack.append(tag)

    try:
      yield
    finally:
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      elapsed = time.time() - start_time
      hierarchical_tag = ".".join(profiler.stack)
      profiler.stack.pop()

      if profiler.enabled:
        record = profiler.records[hierarchical_tag]
        record["call_count"] += 1
        record["total_time"] += elapsed

        # for ancestor_index in range(len(profiler.stack)):
        #     ancestor_tag = ".".join(profiler.stack[:ancestor_index + 1])
        #     profiler.records[ancestor_tag]["cum_time"] += elapsed

        if len(profiler.stack) > 0:
          hierarchical_tag = ".".join(profiler.stack)
          profiler.records[hierarchical_tag]["cum_time"] += elapsed

  def get_dataframe(self):
    for key, record in self.records.items():
      if record["call_count"] > 0 :
        record["leaf_time"] = record["total_time"] - record["cum_time"]

    data = {
        "Tag": [],
        "Calls": [],
        "TotalTime": [],
        "CumTime": [],
        "LeafTime": []
    }

    for tag, record in self.records.items():
      data["Tag"].append(tag)
      data["Calls"].append(record["call_count"])
      data["TotalTime"].append(record["total_time"])
      data["CumTime"].append(record["cum_time"])
      data["LeafTime"].append(record.get("leaf_time", 0.0))

    return pd.DataFrame(data)

  def report(self, sort_by="cum_time", limit=100):
    sorted_records = sorted(self.records.items(), key=lambda x: x[1][sort_by], reverse=True)

    print(f"{'Tag':<80} {'Calls':<10} {'Total Time':<15} {'Cumulative Time':<15}")
    print("=" * 90)
    for tag, data in sorted_records[:limit]:
      print(f"{tag:<80} {data['call_count']:<10} {data['total_time']:<15.13f} {data['cum_time']:<15.13f}")

  def report_under_parent(self, parent_tag):
    prefix = parent_tag + '.'
    for tag in self.records:
      if tag.startswith(prefix):
        print(
          f"{tag:<80} {self.records[tag]['call_count']:<10} {self.records[tag]['total_time']:<15.6f} {self.records[tag]['cum_time']:<15.6f}")

  def clear(self):
    self.records.clear()
    self.stack = []


get_profiler = Profiler.get_instance


@contextmanager
def measure(tag):
  with get_profiler().measure(tag):
    yield


def profile(tag):
  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      with measure(tag):
        return func(*args, **kwargs)
    return wrapper
  return decorator

profile_desc = [
  ("linear", r"linear$", "TotalTime"),
  ("hadamard", [r"(q|k)roate$", r"downproj.matmul_hadU_cuda$"], "TotalTime"),
  ("rope", r"apply_rope$", "LeafTime"),
  ("attn_weights_cal", r"attn_weights_cal$", "TotalTime"),
  ("softmax", r"softmax$", "TotalTime"),
  ("attn_output", r"attn_output$", "TotalTime"),
  ("actfn", r"actfn$", "TotalTime"),
  ("quant_param_find", r"find_params", "TotalTime"),
  ("quant", r"asmy_ste_quantize", "TotalTime"),
  ("embed", r"embed_tokens", "TotalTime"),
  ("rope_param_cal", r"rope_param_cal$", "TotalTime"),
  ("attn_mask_cal", r"attn_mask_cal$", "TotalTime"),
  ("rmsnorm", r"rmsnorm$", "TotalTime"),
  ("mlpmul", r"mlpmul$", "TotalTime"),
  ("reshape", [r"head_reshape$", r"repeat$"], "TotalTime"),
  ("output_reshape", [r"output_reshape$"], "TotalTime"),
]

def get_profiled_df(df):
  profile_data = {
    "name" : [],
    "time" : []
  }
  for (name, pat, col) in profile_desc:
    if isinstance(pat ,list):
      results = []
      for p in pat:
        results.append(df[df["Tag"].str.contains(p)])
      df_ = pd.concat(results)
    else:
      df_ = df[df["Tag"].str.contains(pat)]
    
    time = df_[col].sum()
    profile_data["name"].append(name)
    profile_data["time"].append(time)

  profile_df = pd.DataFrame(profile_data)
  profile_df = profile_df.sort_values(by="time", ascending=False)
  profile_df = profile_df.reset_index(drop=True)

  return profile_df

def plot_profiled_df(df, fname=""):
  ax = df.plot.barh(x='name', y='time', legend=False)
  ax.set_xscale('log')
  ax.set_xlabel("time (log scale)")

  xmax = df['time'].max()
  ax.set_xlim(left=df['time'].min() * 0.9, right=xmax * 1.2)

  # 3) Annotate each bar
  for bar in ax.patches:
      width = bar.get_width()
      y = bar.get_y() + bar.get_height() / 2
      ax.text(
          width * 1.01,  # a little bit to the right of the bar
          y,
          f"{width:4.6f}",  # format with thousands separators
          va='center',
          fontsize=9
      )
  
  if fname:
    plt.savefig(fname, bbox_inches='tight', dpi=300)

  plt.tight_layout()
  plt.show()

def run_profile(model, batch_size, past_seq_len, seq_len, device, fname=""):
  if device == "cuda":
    model = model.cuda()
  elif device == "cpu":
    model = model.cpu()
  else:
    raise ValueError(f"Unknown device {device}")

  # past_seq_len = 0
  # target_seq_len = 32
  # batch_size = 2

  input_ids = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.int64, device=model.device)

  if past_seq_len > 0:
    past_key_values = list()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    kv_shape = (batch_size, model.config.num_key_value_heads, past_seq_len, head_dim)
    for i in range(model.config.num_hidden_layers):
        key = torch.randn(*kv_shape, device=model.device, dtype=torch.float16)
        value = torch.randn(*kv_shape, device=model.device, dtype=torch.float16)
        past_key_values.append([key, value])
  else:
    past_key_values = None


  # warmuup
  print("Warmup...")
  with torch.no_grad():
    for i in range(1):
      output = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)


  # measure
  print("Start profiling...")
  get_profiler().clear()
  with measure("top"):
    with torch.no_grad():
      for i in range(1):
        output = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)


  profiler = get_profiler()
  profiler.report("total_time")
  df = profiler.get_dataframe()
  df = df.sort_values(by="CumTime", ascending=False)
  profile_df = get_profiled_df(df)

  plot_profiled_df(profile_df, fname)

  return profile_df