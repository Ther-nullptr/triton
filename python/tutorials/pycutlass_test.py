import cutlass
import numpy as np

plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.RowMajor)
A, B, C, D = [np.ones((4096, 4096), dtype=np.float16) for i in range(4)]
plan.run(A, B, C, D)