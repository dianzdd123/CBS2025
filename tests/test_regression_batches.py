"""Exercise actual experiment functions without importing data-loading programs.

Torch I/O is replaced with small array-backed fixtures. These tests validate
batch handling and accounting, not Torch models, gradients or performance.
"""
import ast
from contextlib import nullcontext, redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ['depth_baseline_data_size.py', 'piml_depth.py', 'piml_depth_adapt.py', 'piml_depth_adapt_data_size.py']

class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
    def to(self, *_): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def reshape(self, *shape): return Tensor(self.value.reshape(*shape))
    def squeeze(self, *axes): return Tensor(self.value.squeeze(*axes))
    def size(self, axis): return self.value.shape[axis]

class Model:
    def train(self): pass
    def eval(self): pass
    def state_dict(self): return {}
    def __call__(self, images, *_): return images

def load_functions(name):
    tree = ast.parse((ROOT / name).read_text())
    # Avoid top-level experiment setup, filesystem access and training at import.
    module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'train_model', 'evaluate', 'test_model', 'brenner'}], type_ignores=[])
    namespace = {'np': np, 'g_s':(6,6), 'device':'cpu', 'torch': SimpleNamespace(no_grad=nullcontext, save=MagicMock()),
                 'mean_squared_error': lambda a,b: float(np.mean((np.array(a)-np.array(b))**2)),
                 'r2_score': lambda a,b: float(1-np.sum((np.array(a)-np.array(b))**2)/np.sum((np.array(a)-np.mean(a))**2))}
    exec(compile(module, str(ROOT / name), 'exec'), namespace)
    return namespace

def batches(name):
    items = [(Tensor([[1.],[2.]]), Tensor([1.,2.])), (Tensor([[3.]]), Tensor([3.]))]
    if name.startswith('piml'):
        # Physics scripts unpack image, features and depth in that order.
        return [(x, Tensor([[0.]]*x.size(0)), y) for x,y in items]
    return items

class RegressionBatches(unittest.TestCase):
    def test_final_single_sample_is_evaluated(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                ns=load_functions(name)
                self.assertEqual(ns['evaluate'](Model(), batches(name)), 0.)
                self.assertEqual(ns['test_model'](Model(), batches(name)), (0.,1.))

    def test_mean_training_loss_uses_sample_counts(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                ns=load_functions(name)
                losses=iter([2.,5.])
                def criterion(*_):
                    value=next(losses)
                    return SimpleNamespace(item=lambda:value, backward=lambda:None)
                output=io.StringIO()
                with redirect_stdout(output):
                    ns['train_model'](Model(), batches(name), batches(name), MagicMock(), criterion, 'unused', epochs=1)
                self.assertIn('Mean training loss: 3.0000', output.getvalue())

    def test_empty_training_loader_fails_clearly(self):
        for name in SCRIPTS:
            with self.subTest(script=name), self.assertRaisesRegex(ValueError, 'no batches'):
                load_functions(name)['train_model'](Model(), [], batches(name), MagicMock(), MagicMock(), 'unused', epochs=1)

    def test_brenner_uses_signed_float_arithmetic(self):
        ns=load_functions('local_features_extraction_for_microrobot_depth_estimation_.py')
        self.assertEqual(ns['brenner'](np.array([[255,0,0]], dtype=np.uint8)), 65025.)
